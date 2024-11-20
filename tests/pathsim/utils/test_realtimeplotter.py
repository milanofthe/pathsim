########################################################################################
##
##                                     TESTS FOR 
##                             'utils/realtimeplotter.py'
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from pathsim.utils.realtimeplotter import RealtimePlotter


# TESTS ================================================================================

class TestRealtimePlotter(unittest.TestCase):
    """
    test the implementation of the 'RealtimePlotter' class 
    """

    def setUp(self):
        self.max_samples = 100
        self.update_interval = 0.5
        self.labels = ['Test 1', 'Test 2']
        self.x_label = 'X Axis'
        self.y_label = 'Y Axis'

    @patch('pathsim.utils.realtimeplotter.plt')
    def test_init(self, mock_plt):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plotter = RealtimePlotter(
            max_samples=self.max_samples,
            update_interval=self.update_interval,
            labels=self.labels,
            x_label=self.x_label,
            y_label=self.y_label
        )

        self.assertEqual(plotter.max_samples, self.max_samples)
        self.assertEqual(plotter.update_interval, self.update_interval)
        self.assertEqual(plotter.labels, self.labels)
        self.assertEqual(plotter.x_label, self.x_label)
        self.assertEqual(plotter.y_label, self.y_label)

        mock_plt.subplots.assert_called_once()
        mock_ax.set_xlabel.assert_called_once_with(self.x_label)
        mock_ax.set_ylabel.assert_called_once_with(self.y_label)
        mock_ax.grid.assert_called_once_with(True)

    @patch('pathsim.utils.realtimeplotter.plt')
    @patch('pathsim.utils.realtimeplotter.time')
    def test_update_all(self, mock_time, mock_plt):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_line = MagicMock()
        mock_ax.plot.return_value = [mock_line]
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_time.time.return_value = 0

        plotter = RealtimePlotter(labels=self.labels)

        x = np.array([1, 2, 3])
        y = np.array([[1, 2, 3], [4, 5, 6]])

        result = plotter.update_all(x, y)

        self.assertTrue(result)
        self.assertTrue(hasattr(plotter, 'data'))
        self.assertTrue(hasattr(plotter, 'lines'))

    @patch('pathsim.utils.realtimeplotter.plt')
    @patch('pathsim.utils.realtimeplotter.time')
    def test_update(self, mock_time, mock_plt):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_line = MagicMock()
        mock_ax.plot.return_value = [mock_line]
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_time.time.return_value = 0

        plotter = RealtimePlotter(max_samples=5)

        for i in range(10):
            result = plotter.update(i, i)
            self.assertTrue(result)

        self.assertTrue(hasattr(plotter, 'data'))
        self.assertTrue(hasattr(plotter, 'lines'))

    @patch('pathsim.utils.realtimeplotter.plt')
    def test_on_close(self, mock_plt):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plotter = RealtimePlotter()
        plotter.on_close(None)
        self.assertFalse(plotter.is_running)

    @patch('pathsim.utils.realtimeplotter.plt')
    def test_show(self, mock_plt):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plotter = RealtimePlotter()
        plotter.show()
        mock_plt.show.assert_called_with(block=False)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)