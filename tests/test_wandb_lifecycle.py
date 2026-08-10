import sys
import types
import unittest
from unittest import mock

import benchmark_spec_kv_quantization as benchmark


class WandbLifecycleTest(unittest.TestCase):
    def test_explicit_finish_unregisters_atexit_and_tears_down_service(self):
        run = mock.Mock()
        handler = mock.Mock()
        fake_wandb = types.SimpleNamespace(teardown=mock.Mock())
        benchmark._WANDB_FINISH_HANDLERS[id(run)] = handler

        with (
            mock.patch.object(benchmark.atexit, "unregister") as unregister,
            mock.patch.dict(sys.modules, {"wandb": fake_wandb}),
        ):
            benchmark.finish_wandb(run)

        unregister.assert_called_once_with(handler)
        run.finish.assert_called_once_with()
        fake_wandb.teardown.assert_called_once_with()
        self.assertNotIn(id(run), benchmark._WANDB_FINISH_HANDLERS)

    def test_none_run_is_a_noop(self):
        benchmark.finish_wandb(None)

    def test_slurm_success_bypasses_broken_native_teardown(self):
        with (
            mock.patch.dict(
                benchmark.os.environ,
                {"SLURM_JOB_ID": "123", "KV_REDUCE_HARD_EXIT_AFTER_SUCCESS": "1"},
                clear=True,
            ),
            mock.patch.object(benchmark.sys.stdout, "flush") as stdout_flush,
            mock.patch.object(benchmark.sys.stderr, "flush") as stderr_flush,
            mock.patch.object(benchmark.os, "_exit") as hard_exit,
        ):
            benchmark.hard_exit_after_success()

        stdout_flush.assert_called_once_with()
        stderr_flush.assert_called_once_with()
        hard_exit.assert_called_once_with(0)

    def test_hard_exit_can_be_disabled_on_slurm(self):
        with (
            mock.patch.dict(
                benchmark.os.environ,
                {"SLURM_JOB_ID": "123", "KV_REDUCE_HARD_EXIT_AFTER_SUCCESS": "0"},
                clear=True,
            ),
            mock.patch.object(benchmark.os, "_exit") as hard_exit,
        ):
            benchmark.hard_exit_after_success()

        hard_exit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
