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


if __name__ == "__main__":
    unittest.main()
