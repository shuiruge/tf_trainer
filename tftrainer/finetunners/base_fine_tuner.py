#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script involves `BaseFineTuner`"""


import abc
from collections import deque



class BaseFineTuner(abc.ABC):
    """Abstract base class of fine-tuner that makes fine-tuning.

    Definition:
        A "fine-tuner" is an object that dynamically fine-tunes the parameters in
        a running iterative model, basing on the temperal LOCAL STATE, i.e. the
        performances in the several recent iterations; and the fine-tuning can
        be a trial, as which a backward-rolling shall be provided as long as the
        fine-tuning (as a trial) is judged as failed; and in determing its next
        move, this failure is to be taken into account.

    Abbrivations:
        "ft" for "fine-tuning".

    Args:
        n_recent_iters:
            `int`, as the number of the several recent iterations.

        n_trials:
            `int`, as the number of iteration of trials of observing once an
            operation of fine-tuning is taken.

        max_n_rollbacks:
            `int` or `None`, as the maximum number of continous rolling-backs
            that can be taken.

    Methods:
        ft:
            The main method to be called in the context manager, which returns
            the fine-tuning parameter basing on the temporal LOCAL STATE of
            iteration, or `None` if no fine-tuning is needed or under observing
            after the previous fine-tuning.

    Raises:
        Exception:
            When the number of continous rolling-backs reaches the
            `self.max_n_rollbacks` (if not `None`).
    """


    def __init__(self, n_recent_iters, n_trials, max_n_rollbacks=None):

        self.n_recent_iters = n_recent_iters
        self.n_trials = n_trials
        self.max_n_rollbacks = max_n_rollbacks

        # Initially, we shall let the iteration go by itself.
        self.under_observing = False

        # Used for counting how many continous rolling-backs.
        self.n_rollbacks = 0


    def __enter__(self):

        self.recent_iteration_logs = deque(maxlen=self.n_recent_iters)
        self.trial_logs = deque(maxlen=self.n_trials)
        self.ft_parameter = None
        self.rollback_point = None
        self.failed_ft_parameter = None


    def __exit__(self):

        # Release the memory
        del self.recent_iteration_logs
        del self.trial_logs
        del self.ft_parameter
        del self.rollback_point
        del self.failed_ft_parameter


    @abc.abstractmethod
    def needs_ft(self):
        """Abstract method. Determines whether fine-tuning is needed in temporal
        state or not, basing on `self.recent_iteration_logs`.

        Returns:
            `bool`.
        """
        pass


    @abc.abstractmethod
    def get_rollback_point(self):
        """Abstract method. Gets the rollback point (i.e. `self.rollback_point`)
        for backward-rolling if the fine-tuning (as a trial) is judged as failed.

        This method is to be implemented after being instantated, since it
        depends on the model under iteration.

        Returns:
            The type of `self.rollback_point` that is used by `self.rollback()`,
            thus is implementation dependent.
        """
        pass


    def reset_rollback_point(self):
        """Reset the rollback point (i.e. `self.rollback_point`) for backward-
        rolling if the fine-tuning (as a trial) is judged as failed."""
        self.rollback_point = self.get_rollback_point()


    @abc.abstractmethod
    def is_proper_ft(self):
        """Abstract method. Makes judgement whether a fine-tuning is proper or
        not, basing on `self.trial_logs` and `self.recent_iteration_logs` before
        merged with `self.trial_logs` by `self.merge_logs()`.

        Returns:
            `bool`.
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


    def log_failure(self):
        """Logs the failure, for making the next fine-tuning operation after
        rolling back to the rollback-point."""
        self.failed_ft_parameter = self.ft_parameter


    @abc.abstractmethod
    def get_ft_parameter(self):
        """Abstract method. Makes determination of which direction shall the fine-
        tuning go along, basing on temporal attributes `self.recent_iteration_logs`
        and `self.failed_ft_parameter` (if not `None`).

        Returns:
            `dict` object as the `kwargs` of the running iterative model for its
            parameters.
        """
        pass


    def update_recent_iteration_logs(self, parameter, performance):
        """Update the parameters and performances in `self.recent_iteration_logs`
        by the result of the MOST recent iteration.

        Args:
            parameter:
                `dict` object as the `kwargs` of the running iterative model for
                its parameters.

            performance:
                `Any`, implementation dependent.
        """
        self.recent_iteration_logs.append((parameter, performance))


    def update_trial_logs(self, parameter, performance):
        """Update the parameters and performances in `self.trial_logs` by the
        result of the MOST recent iteration. This methid is called only when
        `self.under_observing` is `True`.

        Args:
            parameter:
                `dict` object as the `kwargs` of the running iterative model for
                its parameters.

            performance:
                `Any`, implementation dependent.
        """
        self.trial_logs.append((parameter, performance))


    def stop_observing(self):
        """Check whether we shall stop observing or not. This method is to be
        called only when `self.under_observing` is `True`.

        Returns:
            `bool`.
        """
        return self.trial_logs.count() == self.n_trials


    def merge_logs(self):
        """Merges `self.trial_logs` to `self.recent_iteration_logs` and then
        clear `self.trial_logs`."""

        for log in self.trial_logs:
            self.recent_iteration_logs.append(log)

        self.trial_logs.clear()


    def ft(self, parameter, performance):
        """The main method to be called in the context manager, which returns
        the fine-tuning parameter basing on the temporal LOCAL STATE of
        iteration, or `None` if no fine-tuning is needed or under observing
        after the previous fine-tuning.

        Args:
            parameter:
                `dict` object as the `kwargs` of the running iterative model for
                its parameters.

            performance:
                `Any`, implementation dependent.

        Returns:
            `dict` object as the `kwargs` of the running iterative model for its
            parameters, or `None` if there's no need of fine-tuning in the
            temperal iteration.
        """

        if self.under_observing:

            self.update_trial_logs(parameter, performance)

            if self.stop_observing():

                if self.is_proper_ft():
                    self.merge_logs()

                    # Re-initialize `self.n_rollbacks`
                    self.n_rollbacks = 0

                    # Pass the the next iteration
                    return None

                else:
                    self.log_failure()
                    self.rollback()
                    self.n_rollbacks += 1

                    if self.max_n_rollbacks is not None \
                           and self.n_rollbacks == self.max_n_rollbacks:
                        except_msg = 'The maximum number of continous ' \
                                   + 'rolling-backs has been reached.'
                        raise Exception(except_msg)

                    # Pass to the calling of `self.get_ft_parameter()` in the end
                    pass

            else:
                # Pass the the next iteration
                return None

        else:
            self.update_recent_iteration_logs(parameter, performance)

            if not self.needs_ft():
                # Pass the the next iteration
                return None

        # If has not returned (`None`) yet
        new_ft_parameter = self.get_ft_parameter()

        # and re-initialize the `self.failed_ft_parameter`
        self.failed_ft_parameter = None
        # and starting observing the new operation of fine-tuning to see if it
        # is proper
        self.under_observing = True

        return new_ft_parameter
