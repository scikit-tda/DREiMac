import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class PlotUtils:
    """
    Plotting utilities for DREiMac's output.

    """

    @staticmethod
    def imscatter(X, P, dim, zoom=1, ax=None):
        """
        Plot patches in specified locations in R2

        Parameters
        ----------
        X : ndarray (N, 2)
            The positions of each patch in R2
        P : ndarray (N, dim*dim)
            An array of all of the patches
        dim : int
            The dimension of each patch
        ax : matplotlib axes, optional
            If given, plot on those axes, otherwise plot
            on current axes by calling gca()

        """
        # https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points

        ax = ax or plt.gca()
        min_color = np.min(P)
        max_color = np.max(P)
        for i in range(P.shape[0]):
            patch = np.reshape(P[i, :], (dim, dim))
            x, y = X[i, :]
            im = OffsetImage(patch, zoom=zoom, cmap="gray")
            im.get_children()[0].set_clim(vmin=min_color, vmax=max_color)
            ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)
            ax.add_artist(ab)
        ax.update_datalim(X)
        ax.autoscale()
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    @staticmethod
    def plot_patches(P, zoom=1, ax=None):
        """
        Plot patches in a best fitting rectangular grid

        Parameters
        ----------
        ax : matplotlib axes, optional
            If given, plot on those axes, otherwise plot
            on current axes by calling gca()

        """
        N = P.shape[0]
        d = int(np.sqrt(P.shape[1]))
        dgrid = int(np.ceil(np.sqrt(N)))
        ex = np.arange(dgrid)
        x, y = np.meshgrid(ex, ex)
        X = np.zeros((N, 2))
        X[:, 0] = x.flatten()[0:N]
        X[:, 1] = y.flatten()[0:N]
        return PlotUtils.imscatter(X, P, d, zoom, ax)

    @staticmethod
    def plot_proj_boundary(ax=None):
        """
        Plot the boundary of RP2 on the unit circle

        Parameters
        ----------
        ax : matplotlib axes, optional
            If given, plot on those axes, otherwise plot
            on current axes by calling gca()

        """

        ax = ax or plt.gca()
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), "c")
        ax.axis("equal")
        ax.arrow(
            -0.1, 1, 0.001, 0, head_width=0.15, head_length=0.2, fc="c", ec="c", width=0
        )
        ax.arrow(
            0.1,
            -1,
            -0.001,
            0,
            head_width=0.15,
            head_length=0.2,
            fc="c",
            ec="c",
            width=0,
        )
        ax.set_facecolor((0.35, 0.35, 0.35))
        return ax

    @staticmethod
    def plot_2sphere_boundary(ax=None):
        """
        Plot the boundary of two sphere hemispheres

        Parameters
        ----------
        ax : matplotlib axes, optional
            If given, plot on those axes, otherwise plot
            on current axes by calling gca()

        """

        second_circle_offset = 2.5
        ax = ax or plt.gca()
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), "c")
        ax.plot(np.cos(t) + second_circle_offset, np.sin(t), "c")
        ax.axis("equal")

        ax.arrow(
            0, 0.9, 0, 0.001, head_width=0.15, head_length=0.2, fc="c", ec="c", width=0
        )
        ax.arrow(
            second_circle_offset,
            1.1,
            0,
            -0.001,
            head_width=0.15,
            head_length=0.2,
            fc="c",
            ec="c",
            width=0,
        )

        ax.arrow(
            0,
            -0.9,
            0,
            -0.001,
            head_width=0.15,
            head_length=0.2,
            fc="c",
            ec="c",
            width=0,
        )
        ax.arrow(
            second_circle_offset,
            -1.1,
            0,
            0.001,
            head_width=0.15,
            head_length=0.2,
            fc="c",
            ec="c",
            width=0,
        )

        ax.arrow(
            0.9, 0, 0.001, 0, head_width=0.15, head_length=0.2, fc="c", ec="c", width=0
        )
        ax.arrow(
            second_circle_offset + 1.1,
            0,
            -0.001,
            0,
            head_width=0.15,
            head_length=0.2,
            fc="c",
            ec="c",
            width=0,
        )

        ax.arrow(
            -0.9,
            0,
            -0.001,
            0,
            head_width=0.15,
            head_length=0.2,
            fc="c",
            ec="c",
            width=0,
        )
        ax.arrow(
            second_circle_offset - 1.1,
            0,
            0.001,
            0,
            head_width=0.15,
            head_length=0.2,
            fc="c",
            ec="c",
            width=0,
        )

        ax.set_facecolor((0.35, 0.35, 0.35))

        return ax

    @staticmethod
    def plot_3sphere_mesh(n_parallels = 10, n_meridians = 20, alpha=0.1, ax=None):
        # TODO: docstring

        ax = ax or plt.gca()
        u, v = np.mgrid[0 : 2 * np.pi : n_meridians * 1j, 0 : np.pi : n_parallels * 1j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="black", alpha=alpha)
        return ax

    @staticmethod
    def set_axes_equal(ax):
        # taken from https://stackoverflow.com/a/31364297/2171328
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Parameters
        ----------
          ax: a matplotlib axis, e.g., as output from plt.gca().

        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        return ax

    @staticmethod
    def plot_rp2_stereo(S, f, arrowcolor="c", facecolor=(0.15, 0.15, 0.15)):
        """
        Plot a 2D Stereographic Projection

        Parameters
        ----------
        S : ndarray (N, 2)
            An Nx2 array of N points to plot on RP2
        f : ndarray (N) or ndarray (N, 3)
            A function with which to color the points, or a list of colors

        """

        def _plot_rp2_circle(
            ax, arrowcolor="c", facecolor=(0.15, 0.15, 0.15), do_arrows=True, pad=1.1
        ):
            """
            Plot a circle with arrows showing the identifications for RP2.
            Set an equal aspect ratio and get rid of the axis ticks, since
            they are clear from the circle

            Parameters
            ----------
            ax: matplotlib axis
                Axis onto which to plot the circle+arrows
            arrowcolor: string or ndarray(3) or ndarray(4)
                Color for the circle and arrows
            facecolor: string or ndarray(3) or ndarray(4)
                Color for background of the plot
            do_arrows: boolean
                Whether to draw the arrows
            pad: float
                The dimensions of the window around the unit square

            """
            t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(t), np.sin(t), c=arrowcolor)
            ax.axis("equal")
            ax = plt.gca()
            if do_arrows:
                ax.arrow(
                    -0.1,
                    1,
                    0.001,
                    0,
                    head_width=0.15,
                    head_length=0.2,
                    fc=arrowcolor,
                    ec=arrowcolor,
                    width=0,
                )
                ax.arrow(
                    0.1,
                    -1,
                    -0.001,
                    0,
                    head_width=0.15,
                    head_length=0.2,
                    fc=arrowcolor,
                    ec=arrowcolor,
                    width=0,
                )
            ax.set_facecolor(facecolor)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([-pad, pad])
            ax.set_ylim([-pad, pad])

        if not (S.shape[1] == 2):
            warnings.warn(
                "Plotting stereographic RP2 projection, but points are not 2 dimensional"
            )
        _plot_rp2_circle(plt.gca(), arrowcolor, facecolor)
        if f.size > S.shape[0]:
            plt.scatter(S[:, 0], S[:, 1], 20, c=f, cmap="afmhot")
        else:
            plt.scatter(S[:, 0], S[:, 1], 20, f, cmap="afmhot")

    @staticmethod
    def plot_rp3_stereo(ax, S, f, draw_sphere=False):
        """
        Plot a 3D Stereographic Projection

        Parameters
        ----------
        ax : matplotlib axis
            3D subplotting axis
        S : ndarray (N, 3)
            An Nx3 array of N points to plot on RP3
        f : ndarray (N) or ndarray (N, 3)
            A function with which to color the points, or a list of colors
        draw_sphere : boolean
            Whether to draw the 2-sphere

        """
        if not (S.shape[1] == 3):
            warnings.warn(
                "Plotting stereographic RP3 projection, but points are not 4 dimensional"
            )
        if f.size > S.shape[0]:
            ax.scatter(S[:, 0], S[:, 1], S[:, 2], c=f, cmap="afmhot")
        else:
            c = plt.get_cmap("afmhot")
            C = f - np.min(f)
            C = C / np.max(C)
            C = c(np.array(np.round(C * 255), dtype=np.int32))
            C = C[:, 0:3]
            ax.scatter(S[:, 0], S[:, 1], S[:, 2], c=C, cmap="afmhot")
        if draw_sphere:
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 20j]
            ax.set_aspect("equal")
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v)
            ax.plot_wireframe(x, y, z, color="k")
