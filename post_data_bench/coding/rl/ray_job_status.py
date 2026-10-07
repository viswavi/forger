"""Ray's CLI can return zero for STOPPED; require a successful job explicitly."""
import sys


def require_success(status):
    """Reject any Ray terminal status other than successful completion."""
    if getattr(status, "value", status) != "SUCCEEDED":
        raise SystemExit("Final RL job did not reach SUCCEEDED")


if __name__ == "__main__":
    from ray.job_submission import JobSubmissionClient

    require_success(
        JobSubmissionClient("http://127.0.0.1:8265").get_job_status(sys.argv[1])
    )
