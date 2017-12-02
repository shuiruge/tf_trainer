#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script involves `BaseFineTuner`"""


import abc
from base_fine_tuner import BaseFineTuner


class TFFineTuner(BaseFineTuner):
    """Abstract base class of fine-tuner of TensorFlow."""


    def __init__(self, saver, *args, **kwargs

        self.saver = saver

        super().__init__(*args, **kwargs)


    def get_rollback_point(self):
        """Gets the rollback point (i.e. `self.rollback_point`) for backward-
        rolling if the fine-tuning (as a trial) is judged as failed.

        This method is to be implemented after being instantated, since it
        depends on the model under iteration.

        Returns:
            The type of `self.rollback_point` that is used by `self.rollback()`,
            thus is implementation dependent.
        """
        pass


    @abc.abstractmethod
    def rollback(self):
        """Abstract method. Restores from `self.rollback_point` once the fine-
        tuning operation is judged as failed by `self.is_proper_ft()`.

        This method is to be implemented after being instantated, since it
        depends on the model under iteration.
        """
        pass


