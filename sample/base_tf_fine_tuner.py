#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script involves `BaseFineTuner`"""


from base_fine_tuner import BaseFineTuner


class BaseTensorFlowFineTuner(BaseFineTuner):
    """Abstract base class of fine-tuner of TensorFlow.

    Args:
        saver:
            XXX

        sess:
            XXX

        save_path:
            XXX

        *args, **kwargs:
            XXX
    """


    def __init__(self, saver, sess, save_path,
                 n_recent_iters=10, n_trials=10,
                 max_n_rollbacks=None):

        super().__init__(n_recent_iters, n_trials,
                         max_n_rollbacks=max_n_rollbacks)

        self.saver = saver
        self.sess = sess
        self.save_path = save_path


    def __exit__(self):

        super().__exit__()

        # XXX: Delete the checkpoint of rollback and related files on disk.


    def get_rollback_point(self):
        """Gets the rollback point (i.e. `self.rollback_point`) for backward-
        rolling if the fine-tuning (as a trial) is judged as failed.

        Returns:
            `str`, as the `save_path` of the checkpoint of rollback on disk.
        """
        rollback_save_path = \
            os.path.join(os.path.dirname(self.save_path), 'rollback')
        self.saver.save(self.sess, rollback_save_path)


    def rollback(self):
        """Restores from `self.rollback_point` once the fine-tuning operation is
        judged as failed by `self.is_proper_ft()`.
        """
        rollback_save_path = self.rollback_point
        self.saver.restore(self.sess, rollback_save_path)


    def merge_logs(self):
        """Override."""
        super().merge_logs()
        # Update summaries.
