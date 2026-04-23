"""AST-based pipeline compiler.

``compile_function(fn)`` parses fn's source and interprets it to produce
a DataSpec graph. Control flow (`for`/`if`) operating on DataSpecs is
lowered into FOREACH / COND nodes; everything else executes as normal
Python at compile time.

The function is never re-executed as Python; the AST is the source of truth.
"""
from __future__ import annotations

import ast
import inspect
from typing import Any

from pyforger.core import DataSpec
from pyforger.graph.primitives import cond, foreach
from pyforger.graph.symbolic import ExprSpec, ItemSpec


class TracingScopeError(RuntimeError):
    """Raised for patterns the compiler understands but can't legally capture."""


class _Accumulator:
    __slots__ = ("entries",)

    def __init__(self):
        self.entries: list[DataSpec] = []

    def append(self, value: DataSpec) -> None:
        if not isinstance(value, DataSpec):
            raise TypeError(
                f"Inside a lazy for-loop, list.append must receive a DataSpec "
                f"(got {type(value).__name__})."
            )
        self.entries.append(value)


# ===========================================================================
# Compiler
# ===========================================================================

class _Compiler(ast.NodeVisitor):
    def __init__(self, fn):
        self.fn = fn
        self.globals = dict(fn.__globals__)
        self.closure = {}
        if fn.__closure__:
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                try:
                    self.closure[name] = cell.cell_contents
                except ValueError:
                    raise TracingScopeError(
                        f"free variable {name!r} referenced before assignment "
                        f"in function {fn.__name__!r}"
                    ) from None
        self.scope: dict[str, Any] = {}
        self._result = None

    def compile(self) -> DataSpec:
        if getattr(self.fn, "__name__", "") == "<lambda>":
            raise TypeError("cannot compile a lambda; pass a `def` function")
        try:
            src = inspect.getsource(self.fn)
        except OSError as e:
            raise TracingScopeError(
                f"cannot get source for {self.fn.__name__!r}: {e}. "
                "Functions defined in REPLs or via exec() can't be AST-compiled."
            ) from e
        src = _dedent(src)
        tree = ast.parse(src)
        fn_def = tree.body[0]
        if not isinstance(fn_def, ast.FunctionDef):
            raise TypeError("compile_function requires a plain `def` function")
        for stmt in fn_def.body:
            self._exec_stmt(stmt)
            if self._result is not None:
                break
        if self._result is None:
            raise RuntimeError(
                f"Function {self.fn.__name__!r} did not return a DataSpec"
            )
        return self._result

    def _exec_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                self._fallback_exec(stmt)
                return
            name = stmt.targets[0].id
            val = self._eval_expr(stmt.value)
            self.scope[name] = val
            return

        if isinstance(stmt, ast.AugAssign):
            self._exec_aug_assign(stmt)
            return

        if isinstance(stmt, ast.Return):
            self._result = self._eval_expr(stmt.value) if stmt.value else None
            return

        if isinstance(stmt, ast.For):
            self._exec_for(stmt)
            return

        if isinstance(stmt, ast.If):
            self._exec_top_if(stmt)
            return

        if isinstance(stmt, ast.Expr):
            self._eval_expr(stmt.value)
            return

        if isinstance(stmt, (ast.ImportFrom, ast.Import, ast.Pass,
                             ast.Global, ast.Nonlocal, ast.AnnAssign)):
            self._fallback_exec(stmt)
            return

        raise NotImplementedError(
            f"pyforger.graph compiler: unsupported statement "
            f"{type(stmt).__name__} at line {getattr(stmt, 'lineno', '?')}"
        )

    def _exec_aug_assign(self, stmt: ast.AugAssign) -> None:
        if not isinstance(stmt.target, ast.Name):
            self._fallback_exec(stmt)
            return
        name = stmt.target.id
        if name not in self.scope and name not in self.globals:
            self._fallback_exec(stmt)
            return
        lhs = self.scope.get(name, self.globals.get(name))
        rhs = self._eval_expr(stmt.value)
        if isinstance(lhs, DataSpec) or isinstance(rhs, DataSpec):
            raise TracingScopeError(
                f"augmented assignment on a DataSpec (`{name} {type(stmt.op).__name__} ...`) "
                "is not supported at compile time. Use explicit reassignment instead."
            )
        result = _py_augop(stmt.op, lhs, rhs)
        self.scope[name] = result

    def _exec_top_if(self, stmt: ast.If) -> None:
        test = self._eval_expr(stmt.test)
        if isinstance(test, DataSpec):
            raise TracingScopeError(
                "Top-level `if` on a symbolic value is not supported. "
                "Use `if` inside a for-loop over a DataSpec for per-row routing."
            )
        branch = stmt.body if test else stmt.orelse
        for s in branch:
            self._exec_stmt(s)
            if self._result is not None:
                return

    def _exec_for(self, stmt: ast.For) -> None:
        if not isinstance(stmt.target, ast.Name):
            raise NotImplementedError("for-loop target must be a simple name")
        iter_val = self._eval_expr(stmt.iter)

        if not isinstance(iter_val, DataSpec):
            for value in iter_val:
                saved_has = stmt.target.id in self.scope
                saved = self.scope.get(stmt.target.id)
                self.scope[stmt.target.id] = value
                try:
                    for s in stmt.body:
                        self._exec_stmt(s)
                        if self._result is not None:
                            return
                finally:
                    if saved_has:
                        self.scope[stmt.target.id] = saved
                    else:
                        self.scope.pop(stmt.target.id, None)
            return

        self._compile_symbolic_for(stmt, iter_val)

    def _compile_symbolic_for(self, stmt: ast.For, source: DataSpec) -> None:
        accumulator_name = _find_unique_accumulator(stmt.body)
        if accumulator_name is None:
            raise NotImplementedError(
                "pyforger.graph: for-loop body must use exactly one "
                "`<list>.append(<dataspec>)` statement to emit per-row outputs."
            )
        if accumulator_name not in self.scope or not isinstance(
            self.scope[accumulator_name], list
        ):
            raise NameError(
                f"accumulator list {accumulator_name!r} must be defined (as []) "
                f"before the for-loop."
            )
        outer_accumulator_list = self.scope[accumulator_name]

        captured_globals = self.globals
        captured_closure = self.closure
        captured_scope = dict(self.scope)
        target_id = stmt.target.id
        body_stmts = stmt.body

        def _body_fn(item: ItemSpec) -> DataSpec:
            local_scope = dict(captured_scope)
            local_scope[target_id] = item
            acc = _Accumulator()
            local_scope[accumulator_name] = acc
            sub = _SubCompiler(captured_globals, captured_closure, local_scope,
                               accumulator_name)
            for s in body_stmts:
                sub._exec_body_stmt(s)
            if not acc.entries:
                raise RuntimeError(
                    "for-loop body did not append any DataSpec per iteration"
                )
            if len(acc.entries) == 1:
                return acc.entries[0]
            from pyforger.alter import concat as _concat
            return _concat(acc.entries, name="per_row_concat")

        fe = foreach(source, _body_fn, name=f"foreach_{target_id}")
        outer_accumulator_list.append(fe)

    def _eval_expr(self, expr: ast.expr) -> Any:
        return _eval_expr(expr, self.scope, self.globals, self.closure)

    def _fallback_exec(self, stmt: ast.stmt) -> None:
        if _stmt_touches_dataspec(stmt, self.scope, self.globals, self.closure):
            raise TracingScopeError(
                f"statement {type(stmt).__name__} at line "
                f"{getattr(stmt, 'lineno', '?')} involves a DataSpec but "
                f"the compiler can't handle it. Simplify or use explicit primitives."
            )
        mod = ast.Module(body=[stmt], type_ignores=[])
        code = compile(mod, f"<{self.fn.__name__}>", "exec")
        ns = {**self.globals, **self.closure, **self.scope}
        exec(code, ns)
        for k, v in ns.items():
            if k not in self.globals and k not in self.closure:
                self.scope[k] = v


class _SubCompiler:
    def __init__(self, globals_, closure, scope, accumulator_name):
        self.globals = globals_
        self.closure = closure
        self.scope = scope
        self.accumulator_name = accumulator_name

    def _exec_body_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Expr):
            self._eval_expr(stmt.value)
            return
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                self.scope[stmt.targets[0].id] = self._eval_expr(stmt.value)
                return
            raise NotImplementedError(
                "inside lazy for-body: only single-name Assign supported"
            )
        if isinstance(stmt, ast.If):
            self._exec_if_symbolic(stmt)
            return
        raise NotImplementedError(
            f"inside lazy for-body: unsupported statement "
            f"{type(stmt).__name__} at line {getattr(stmt, 'lineno', '?')}"
        )

    def _exec_if_symbolic(self, stmt: ast.If) -> None:
        test = self._eval_expr(stmt.test)
        from pyforger.alter import concat as _concat

        def _collect_branch(branch: list[ast.stmt]) -> DataSpec:
            branch_scope = dict(self.scope)
            acc = _Accumulator()
            branch_scope[self.accumulator_name] = acc
            branch_sub = _SubCompiler(self.globals, self.closure, branch_scope,
                                       self.accumulator_name)
            for s in branch:
                branch_sub._exec_body_stmt(s)
            if not acc.entries:
                raise RuntimeError("if-branch did not append a DataSpec")
            if len(acc.entries) == 1:
                return acc.entries[0]
            return _concat(acc.entries, name="branch_concat")

        then_spec = _collect_branch(stmt.body)
        else_spec = _collect_branch(stmt.orelse) if stmt.orelse else None

        if isinstance(test, DataSpec):
            if else_spec is None:
                raise NotImplementedError(
                    "symbolic `if` without `else` is not yet supported"
                )
            merged = cond(test, lambda: then_spec, lambda: else_spec, name="if_else")
            self.scope[self.accumulator_name].append(merged)
            return

        # Concrete test: pick branch at compile time.
        branch = stmt.body if test else stmt.orelse
        for s in branch:
            self._exec_body_stmt(s)

    def _eval_expr(self, expr: ast.expr) -> Any:
        return _eval_expr(expr, self.scope, self.globals, self.closure)


# ===========================================================================
# Expression evaluator
# ===========================================================================

def _eval_expr(expr: ast.expr, scope, globals_, closure) -> Any:
    if isinstance(expr, ast.Constant):
        return expr.value
    if isinstance(expr, ast.Name):
        if expr.id in scope:    return scope[expr.id]
        if expr.id in closure:  return closure[expr.id]
        if expr.id in globals_: return globals_[expr.id]
        import builtins
        if hasattr(builtins, expr.id):
            return getattr(builtins, expr.id)
        raise NameError(f"name {expr.id!r} not found")

    if isinstance(expr, ast.Attribute):
        obj = _eval_expr(expr.value, scope, globals_, closure)
        return getattr(obj, expr.attr)

    if isinstance(expr, ast.Subscript):
        obj = _eval_expr(expr.value, scope, globals_, closure)
        key = _eval_expr(expr.slice, scope, globals_, closure)
        return obj[key]

    if isinstance(expr, ast.Compare):
        left = _eval_expr(expr.left, scope, globals_, closure)
        if len(expr.ops) != 1 or len(expr.comparators) != 1:
            raise NotImplementedError("chained comparisons not supported")
        right = _eval_expr(expr.comparators[0], scope, globals_, closure)
        op = expr.ops[0]
        if isinstance(left, DataSpec) or isinstance(right, DataSpec):
            from pyforger.graph.symbolic import _expr as _mkexpr
            op_name = {
                ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le",
                ast.Gt: "gt", ast.GtE: "ge", ast.In: "in",
            }.get(type(op))
            if op_name is None:
                raise NotImplementedError(f"symbolic compare op {type(op).__name__}")
            return _mkexpr(op_name, [left, right])
        return _py_compare(op, left, right)

    if isinstance(expr, ast.Call):
        func = _eval_expr(expr.func, scope, globals_, closure)
        args = [_eval_expr(a, scope, globals_, closure) for a in expr.args]
        kwargs = {kw.arg: _eval_expr(kw.value, scope, globals_, closure)
                  for kw in expr.keywords}
        if callable(func):
            return func(*args, **kwargs)
        raise TypeError(f"object {func!r} is not callable")

    if isinstance(expr, ast.List):
        return [_eval_expr(e, scope, globals_, closure) for e in expr.elts]
    if isinstance(expr, ast.Tuple):
        return tuple(_eval_expr(e, scope, globals_, closure) for e in expr.elts)
    if isinstance(expr, ast.Dict):
        return {
            _eval_expr(k, scope, globals_, closure):
            _eval_expr(v, scope, globals_, closure)
            for k, v in zip(expr.keys, expr.values)
        }

    if isinstance(expr, ast.Lambda):
        return _compile_lambda(expr, scope, globals_, closure)

    if isinstance(expr, ast.JoinedStr):
        parts = []
        for v in expr.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                parts.append(str(_eval_expr(v.value, scope, globals_, closure)))
            else:
                raise NotImplementedError("JoinedStr part")
        return "".join(parts)

    if isinstance(expr, ast.UnaryOp):
        val = _eval_expr(expr.operand, scope, globals_, closure)
        if isinstance(expr.op, ast.Not):
            if isinstance(val, DataSpec):
                from pyforger.graph.symbolic import _expr as _mkexpr
                return _mkexpr("not", [val])
            return not val
        if isinstance(expr.op, ast.USub):
            return -val
        raise NotImplementedError(f"UnaryOp {type(expr.op).__name__}")

    if isinstance(expr, ast.BinOp):
        left = _eval_expr(expr.left, scope, globals_, closure)
        right = _eval_expr(expr.right, scope, globals_, closure)
        if isinstance(left, DataSpec) or isinstance(right, DataSpec):
            from pyforger.graph.symbolic import _expr as _mkexpr
            op_name = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}.get(type(expr.op))
            if op_name is None:
                raise NotImplementedError(f"symbolic BinOp {type(expr.op).__name__}")
            return _mkexpr(op_name, [left, right])
        return _py_binop(expr.op, left, right)

    if isinstance(expr, ast.BoolOp):
        vals = [_eval_expr(v, scope, globals_, closure) for v in expr.values]
        if any(isinstance(v, DataSpec) for v in vals):
            from pyforger.graph.symbolic import _expr as _mkexpr
            op_name = "and" if isinstance(expr.op, ast.And) else "or"
            acc = vals[0]
            for v in vals[1:]:
                acc = _mkexpr(op_name, [acc, v])
            return acc
        if isinstance(expr.op, ast.And):
            for v in vals:
                if not v:
                    return v
            return vals[-1]
        for v in vals:
            if v:
                return v
        return vals[-1]

    raise NotImplementedError(f"expression node {type(expr).__name__}")


def _py_compare(op, left, right) -> bool:
    m = {
        ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b,  ast.LtE:   lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,  ast.GtE:   lambda a, b: a >= b,
        ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
        ast.Is: lambda a, b: a is b, ast.IsNot: lambda a, b: a is not b,
    }
    return m[type(op)](left, right)


def _py_binop(op, left, right):
    import operator as _op
    m = {
        ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
        ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv,
        ast.Mod: _op.mod, ast.Pow: _op.pow,
    }
    return m[type(op)](left, right)


def _py_augop(op, left, right):
    import operator as _op
    m = {
        ast.Add: _op.iadd, ast.Sub: _op.isub, ast.Mult: _op.imul,
        ast.Div: _op.itruediv, ast.FloorDiv: _op.ifloordiv,
    }
    return m[type(op)](left, right)


def _compile_lambda(expr: ast.Lambda, scope, globals_, closure):
    arg_names = [a.arg for a in expr.args.args]

    def _call(*args, **kwargs):
        local = dict(scope)
        for n, val in zip(arg_names, args):
            local[n] = val
        local.update(kwargs)
        return _eval_expr(expr.body, local, globals_, closure)

    return _call


# ===========================================================================
# helpers
# ===========================================================================

def _dedent(src: str) -> str:
    lines = src.splitlines()
    if not lines:
        return src
    for line in lines:
        if line.strip():
            prefix_len = len(line) - len(line.lstrip())
            break
    else:
        return src
    out = []
    for line in lines:
        if line[:prefix_len] == " " * prefix_len or line[:prefix_len] == "\t" * prefix_len:
            out.append(line[prefix_len:])
        else:
            out.append(line)
    return "\n".join(out)


def _find_unique_accumulator(body: list[ast.stmt]) -> str | None:
    """Find THE append target used in this for-body.

    Requires exactly one unique target (errors if multiple different names
    are used as ``<name>.append(...)`` — ambiguous).
    """
    class _Find(ast.NodeVisitor):
        def __init__(self):
            self.names: set[str] = set()

        def visit_Call(self, node):  # noqa: N802
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append"
                    and isinstance(node.func.value, ast.Name)):
                self.names.add(node.func.value.id)
            self.generic_visit(node)

    f = _Find()
    for s in body:
        f.visit(s)
    if len(f.names) == 1:
        return next(iter(f.names))
    if len(f.names) > 1:
        raise TracingScopeError(
            f"for-body uses multiple append targets ({sorted(f.names)}); "
            "only one accumulator per lazy for-loop is supported."
        )
    return None


def _stmt_touches_dataspec(stmt: ast.stmt, scope, globals_, closure) -> bool:
    """Check whether a statement references any Name bound to a DataSpec.

    Uses explicit membership tests rather than `a or b` fallthrough — the
    latter would invoke ``__bool__`` on an ExprSpec and raise TracingError.
    """
    sentinel = object()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in scope:
                v = scope[nm]
            elif nm in closure:
                v = closure[nm]
            elif nm in globals_:
                v = globals_[nm]
            else:
                v = sentinel
            if v is not sentinel and isinstance(v, DataSpec):
                return True
    return False


def compile_function(fn) -> DataSpec:
    """Compile a plain Python function into a DataSpec graph."""
    return _Compiler(fn).compile()
