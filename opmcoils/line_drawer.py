import numpy as np

def get_shifted_line(line_cut, dist):
    """https://math.stackexchange.com/questions/2593627/i-have-a-line-i-want-to-move-the-line-a-certain-distance-away-parallelly."""
    x1, x2, y1, y2 = line_cut[0][0], line_cut[1][0], line_cut[0][1], line_cut[1][1]
    r = np.sqrt((x2 - x1) **2 + (y2 - y1) **2)
    xd = dist / r * (y1 - y2)
    yd = dist / r * (x2 - x1)
    x3 = x1 + xd
    y3 = y1 + yd
    x4 = x2 + xd
    y4 = y2 + yd
    return np.array([[x3, y3], [x4, y4]])


class LineDrawer(object):
    """LineDrawer object."""

    def __init__(self, fig):
        import matplotlib.pyplot as plt

        self.lines = {'main': list(), 'shifted': list(), 'pts': list()}
        self.current_line = None
        self.ax = fig.get_axes()[0]
        self.current_line_data = list()
        self.shift_pressed = False
        self.ctrl_pressed = False

        fig.canvas.mpl_connect('button_press_event',
                               self.on_click)
        fig.canvas.mpl_connect('motion_notify_event',
                               self.on_move)
        fig.canvas.mpl_connect('key_press_event', self.on_press)
        fig.canvas.mpl_connect('key_release_event', self.on_release)   

        plt.title('ctrl (draw), ctrl + shift (horizontal snap), u (undo)')
        plt.show(block=True)

    def _update_figure(self):
        self.ax.figure.canvas.draw()

    def _remove_cut(self):
        # only remove completed cuts
        if len(self.current_line_data) == 1 or len(self.lines['main']) == 0:
            return

        for line_type in ['main', 'shifted', 'pts']:
            self.lines[line_type][-1].remove()
            del self.lines[line_type][-1]
        self._update_figure()

    def on_press(self, event):
        if event.key == 'ctrl+shift':
            self.shift_pressed = True
        if event.key == 'control':
            self.ctrl_pressed = True
        if event.key == 'u':
            self._remove_cut()

    def on_release(self, event):
        if self.shift_pressed and event.key in ('shift', 'ctrl+shift'):
            self.shift_pressed = False
        if self.ctrl_pressed and event.key in ('control', 'ctrl+shift'):
            self.ctrl_pressed = False

    def on_move(self, event):
        """On move."""
        if len(self.current_line_data) != 1 or event.inaxes != self.ax or not self.ctrl_pressed:
            return

        l = self.current_line_data
        x1, y1, x2, y2 = l[0][0], l[0][1], event.xdata, event.ydata

        r = np.sqrt((x2 - x1) **2 + (y2 - y1) **2)
        if r < 0.01:  # avoid warnings of divide by zero
            return

        # Snap to horizontal or vertical
        if self.shift_pressed:
            if abs(event.xdata - x1) > abs(event.ydata - y1):
                y2 = y1 = 0
            else:
                x2 = x1 = 0

        l = get_shifted_line([[x1, y1], [x2, y2]], dist=-10) # -6
        x3, x4, y3, y4 = l[0][0], l[1][0], l[0][1], l[1][1]

        # plot the "in progress" line
        if self.current_line is None:
            line = self.ax.plot([x1, x2], [y1, y2], 'k')
            shifted_line = self.ax.plot([x3, x4], [y3, y4], 'k')
            self.current_line = {'main': line[0], 'shifted': shifted_line[0]}
        else:
            self.current_line['main'].set_xdata([x1, x2])
            self.current_line['main'].set_ydata([y1, y2])
            self.current_line['shifted'].set_xdata([x3, x4])
            self.current_line['shifted'].set_ydata([y3, y4])

        self._update_figure()

    def on_click(self, event):
        """On click."""
        if not self.ctrl_pressed:
            return

        self.current_line_data.append((event.xdata, event.ydata))

        # Save the line.
        if len(self.current_line_data) == 1:
            pts = self.ax.plot(event.xdata, event.ydata, 'r+')[0]
            self.lines['pts'].append(pts)
        elif len(self.current_line_data) == 2:
            self.lines['main'].append(self.current_line['main'])
            self.lines['shifted'].append(self.current_line['shifted'])

            # delete first point and redraw both start/end point
            self.lines['pts'][-1].remove()
            del self.lines['pts'][-1]
            x, y = self.lines['main'][-1].get_data()
            pts = self.ax.plot(x, y, 'r+')[0]
            self.lines['pts'].append(pts)

            self.current_line_data = list()
            self.current_line = None

        self._update_figure()

    def get_line_cuts(self):
        """Get line cuts.

        Returns
        -------
        cuts : array, shape (n_lines, 2, 2)
            The lines. Each line has two points and each
            point has (x, y)
        cuts_shifted : array, shape (n_lines, 2, 2)
            The shifted lines. Each line has two points and each
            point has (x, y)
        """
        cuts = np.zeros((len(self.lines['main']), 2, 2))
        cuts_shifted = cuts.copy()
        for idx, (l1, l2) in enumerate(zip(self.lines['main'],
                                        self.lines['shifted'])):
            cuts[idx] = np.array(l1.get_data()).T
            cuts_shifted[idx] = np.array(l2.get_data()).T
        return cuts, cuts_shifted

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    plt.xlim((0, 600))
    plt.ylim((0, 600))

    ld = LineDrawer(fig)
    cuts, cuts_shifted = ld.get_line_cuts()
