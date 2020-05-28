from mayavi import mlab
import numpy as np
#mlab.options.offscreen = True
import h5py as h5
from tvtk.util.ctf import PiecewiseFunction     # countors in rendering
from tvtk.util.ctf import ColorTransferFunction # colormap for rendering
import matplotlib.pyplot as plt

# first test
def test1():
    X= np.linspace(-10, 10, 100)
    x, y, z = np.meshgrid(X, X, X)
    f = np.cos(0.66 * np.pi * (x) / np.sqrt(x ** 2 + y ** 2 + z ** 2))

    fig = mlab.figure()
    mlab.contour3d(f, contours=6, transparent=True, figure=fig)
    print("sved: {}".format("./test1.png"))
    mlab.show()
    mlab.savefig("./test1.png")
    mlab.clf(fig)
    mlab.close()
#test1()

def reset():
    ''' closes the fig and starts a new one '''
    mlab.clf()
    mlab.close()
    fig = mlab.figure()
    return fig
def test2():

    X = np.linspace(-10, 10, 100)
    x, y, z = np.meshgrid(X, X, X)
    f = np.cos(0.66 * np.pi * (x) / np.sqrt(x ** 2 + y ** 2 + z ** 2))

    fig = mlab.figure()
    mlab.contour3d(f, contours=6, transparent=True, figure=fig)
    print("sved: {}".format("./test1.png"))
    #mlab.show()
    mlab.clf(fig)
    mlab.close()

    fig = reset()       # clear the previos fig
    # Create a scalar field object
    sca = mlab.pipeline.scalar_field(f)     # add the data to the pipeline
    sca.origin = (-10., -10., -10.)         # set the center of the plot
    dx = X[1] - X[0]                        # separation between slices
    sca.spacing = (dx, dx, dx)              # set separation
    sca.scalar_name = 'f'                   # set the name of the field

    #mlab.pipeline.iso_surface(sca, transparent=True, contours=[0., 0.25, 0.5], figure=fig) # plot
    #mlab.show()

    # manually setting opacity for countours
    fig = reset()
    mlab.pipeline.iso_surface(sca, opacity=1., contours=[0.], figure=fig)       # Solid
    mlab.pipeline.iso_surface(sca, opacity=0.4, contours=[0.25], figure=fig)    # transparent
    mlab.pipeline.iso_surface(sca, opacity=0.2, contours=[0.5], figure=fig)     # transparent
    mlab.show()
#test2()

# working with real data
def test4():
    dfile = h5.File('../rho3d.h5', 'r')
    print(dfile.attrs.keys())
    print(dfile.attrs['mass ratio'])
    print(dfile['t=3072.'].attrs.keys())
    #
    dset = dfile['t=3072.']
    xyz = dset.attrs['grid']
    dx = dset.attrs['dx']
    #
    fig = reset()
    mlab.contour3d(dset[:].T, contours=6, transparent=True, figure=fig)
    #mlab.show()

    # see the data limits

    print(np.max(dset[:]), np.min(dset[:]))
    rho_cgs = dset[:] * 6.176269145886166e+17 # convert to cgs
    print("rho_min = %.3e g/cm^3,\nrho_max = %.3e g/cm^3" % (np.max(rho_cgs), np.min(rho_cgs)))
    rho_cgs = np.log10(rho_cgs)

    fig = reset()
    # Create a scalar field object
    scr = mlab.pipeline.scalar_field(rho_cgs.T)
    scr.origin = (-100., -100., 0.)
    dx = dset.attrs['dx']
    scr.spacing = (dx[0], dx[1], dx[2])
    scr.scalar_name = 'rho'

    mlab.pipeline.iso_surface(scr, opacity=1., contours=[13],  figure=fig)
    mlab.pipeline.iso_surface(scr, opacity=0.4, contours=[10], figure=fig)
    mlab.pipeline.iso_surface(scr, opacity=0.2, contours=[8],  figure=fig)
    mlab.show()
#test4()

# volume rendering
def test5():

    X = np.linspace(-10, 10, 100)
    x, y, z = np.meshgrid(X, X, X)
    f = np.cos(0.66 * np.pi * (x) / np.sqrt(x ** 2 + y ** 2 + z ** 2))

    fig = reset()
    sc = mlab.pipeline.scalar_field(f)
    sc.origin = (-10., -10., -10.)
    dx = X[1] - X[0]
    sc.spacing = (dx, dx, dx)
    sc.scalar_name = 'f_xyz'
    mlab.pipeline.volume(sc, vmin=-0.1, vmax=0.6, figure=fig)
    mlab.show()
#test5()

# volume rend. with values
def test6():
    X = np.linspace(-10, 10, 100)
    x, y, z = np.meshgrid(X, X, X)
    f = np.cos(0.66 * np.pi * (x) / np.sqrt(x ** 2 + y ** 2 + z ** 2))

    fig = reset()
    # Create an array of samples between the min and max values we want to show
    smpl = np.linspace(0.0, 0.5, 50)
    # Initiate opacities
    opac = np.zeros_like(smpl)

    # Now, add gaussian-shaped function around the values we are interested into
    centers = [0.0, 0.25, 0.49]
    opacs =   [1.0, 0.5, 0.3]
    widths =  [0.01, 0.04, 0.01]
    for c, o, w in zip(centers, opacs, widths):
        opac += o * np.exp(-((smpl - c) / w) ** 2)

    # Now define piecewise opacity transfer function
    otf = PiecewiseFunction()
    for v, o in zip(smpl, opac):
        otf.add_point(v, o)

    def return_vrend(f, X, fig, otf):
        sc = mlab.pipeline.scalar_field(f)
        sc.origin = (-10., -10., -10.)
        dx = X[1] - X[0]
        sc.spacing = (dx, dx, dx)
        sc.scalar_name = 'logf_xyz'
        vol = mlab.pipeline.volume(sc, vmin=0., vmax=0.52, figure=fig)
        vol._otf = otf
        vol._volume_property.set_scalar_opacity(otf)

        return vol

    #
    return_vrend(f, X, fig, otf)
    mlab.show()
#test6()

# volume rend. with val and change of colormap -- see the GREEN
def test7():

    ''''''
    fig = reset()

    X = np.linspace(-10, 10, 100)
    x, y, z = np.meshgrid(X, X, X)
    f = np.cos(0.66 * np.pi * (x) / np.sqrt(x ** 2 + y ** 2 + z ** 2))

    ''' --- create opacities --- '''

    # Create an array of samples between the min and max values we want to show
    smpl = np.linspace(0.0, 0.5, 50)
    # Initiate opacities
    opac = np.zeros_like(smpl)

    # Now, add gaussian-shaped function around the values we are interested into
    centers = [0.0, 0.25, 0.49]
    opacs = [1.0, 0.5, 0.3]
    widths = [0.01, 0.04, 0.01]
    for c, o, w in zip(centers, opacs, widths):
        opac += o * np.exp(-((smpl - c) / w) ** 2)

    # Now define piecewise opacity transfer function
    otf = PiecewiseFunction()
    for v, o in zip(smpl, opac):
        otf.add_point(v, o)

    ''' --- create colormaps --- '''

    # Initialize the color transfer function and set the range
    ctf = ColorTransferFunction()
    ctf.range = [0., 0.5]

    # Choose a color map and sample it
    cm = plt.get_cmap('jet_r', 10)
    ik = np.arange(0, 10)

    # colors
    ck = cm(ik)[:, :3] # [:, r, g, b]

    # vertexes
    vk = ik / float(ik[-1])
    clrs = [(v, tuple(c)) for v, c in zip(vk, ck)]

    for v, (r, g, b) in clrs:
        ctf.add_rgb_point(0.0 + v * (0.5 - 0.0), r, g, b)

    def return_vrend(f, fig, otf, ctf):
        sc = mlab.pipeline.scalar_field(f)
        sc.origin = (-10., -10., -10.)
        dx = X[1] - X[0]
        sc.spacing = (dx, dx, dx)
        sc.scalar_name = 'logf_xyz'
        vol = mlab.pipeline.volume(sc, vmin=0., vmax=0.52, figure=fig)
        vol._otf = otf
        vol._volume_property.set_scalar_opacity(otf)

        vol._volume_property.set_color(ctf)
        vol._ctf = ctf
        vol.update_ctf = True

        return vol

    return_vrend(f, fig, otf, ctf)
    mlab.show()
    mlab.clf()
    mlab.close()
#test7()

# figure manipulation
def test8():

    X = np.linspace(-10, 10, 100)
    x, y, z = np.meshgrid(X, X, X)
    f = np.cos(0.66 * np.pi * (x) / np.sqrt(x ** 2 + y ** 2 + z ** 2))

    ''' --- create opacities --- '''

    # Create an array of samples between the min and max values we want to show
    smpl = np.linspace(0.0, 0.5, 50)
    # Initiate opacities
    opac = np.zeros_like(smpl)

    # Now, add gaussian-shaped function around the values we are interested into
    centers = [0.0, 0.25, 0.49]
    opacs = [1.0, 0.5, 0.3]
    widths = [0.01, 0.04, 0.01]
    for c, o, w in zip(centers, opacs, widths):
        opac += o * np.exp(-((smpl - c) / w) ** 2)

    # Now define piecewise opacity transfer function
    otf = PiecewiseFunction()
    for v, o in zip(smpl, opac):
        otf.add_point(v, o)

    ''' --- create colormaps --- '''

    # Initialize the color transfer function and set the range
    ctf = ColorTransferFunction()
    ctf.range = [0., 0.5]

    # Choose a color map and sample it
    cm = plt.get_cmap('jet_r', 10)
    ik = np.arange(0, 10)

    # colors
    ck = cm(ik)[:, :3]  # [:, r, g, b]

    # vertexes
    vk = ik / float(ik[-1])
    clrs = [(v, tuple(c)) for v, c in zip(vk, ck)]

    for v, (r, g, b) in clrs:
        ctf.add_rgb_point(0.0 + v * (0.5 - 0.0), r, g, b)

    def return_vrend(f, fig, otf, ctf):
        sc = mlab.pipeline.scalar_field(f)
        sc.origin = (-10., -10., -10.)
        dx = X[1] - X[0]
        sc.spacing = (dx, dx, dx)
        sc.scalar_name = 'logf_xyz'
        vol = mlab.pipeline.volume(sc, vmin=0., vmax=0.52, figure=fig)
        vol._otf = otf
        vol._volume_property.set_scalar_opacity(otf)

        vol._volume_property.set_color(ctf)
        vol._ctf = ctf
        vol.update_ctf = True

        return vol

    ''' --- plot --- '''

    fig = mlab.figure(size=(1378, 720), bgcolor=(0., 0., 0.), fgcolor=(1., 1., 1.))
    vol = return_vrend(f, fig, otf, ctf)
    mlab.orientation_axes(figure=fig)
    mlab.show()
#test8()

# real data
def test9():
    """

    :return:
    """


    # --- loading data ---
    dfile = h5.File('../rho3d.h5', 'r')
    print(dfile.attrs.keys())
    print(dfile.attrs['mass ratio'])
    print(dfile['t=3072.'].attrs.keys())
    #
    dset = dfile['t=3072.']
    xyz = dset.attrs['grid']
    dx = dset.attrs['dx']
    rho_cgs = dset[:] * 6.176269145886166e+17

    ''' plotting '''

    fig = reset()
    sc = mlab.pipeline.scalar_field(rho_cgs.T)
    sc.origin = (-100.,-100.,0.)
    dx = dset.attrs['dx']
    sc.spacing = (dx[0],dx[1],dx[2])
    sc.scalar_name='rho_xyz'

    def get_view(sc, fig):
        im = mlab.pipeline.volume(sc, vmin=6., vmax=13., figure=fig)
        mlab.view(azimuth=45.,
                  elevation=45., distance=600.,
                  focalpoint=[0, 0, 0],
                  figure=None)
        mlab.orientation_axes(figure=fig)
        return im

    get_view(sc, fig)
    mlab.show()
#test9()


def get_ctf(cmap='jet_r', smpls=50, crange=[0.,1.]):
    # Initialize the color transfer function and set the range
    ctf = ColorTransferFunction()
    ctf.range = crange

    # Choose a color map and sample it
    cm  = plt.get_cmap('jet_r', smpls)
    ik  = np.arange(0,smpls)
    # colors
    ck  = cm(ik)[:,:3]
    # vertexes
    vk  = ik / float(ik[-1])
    clrs = [(v,tuple(c)) for v,c in zip(vk, ck)]

    for v, (r,g,b) in clrs:
        ctf.add_rgb_point(crange[0] + v*(crange[1]-crange[0]), r, g, b)
    #
    return ctf

def get_otf(centers, opacs, widths, smpls=50,orange=[0.,1.]):
    # Create an array of samples between the min and max values we want to show
    smpl = np.linspace(orange[0],orange[1],smpls)
    # Initiate opacities
    opac = np.zeros_like(smpl)

    # Now, add gaussian-shaped function around the values we are interested into
    for c,o,w in zip(centers,opacs,widths):
        opac += o * np.exp(-((smpl-c)/w)**2)
    #
    # Now define piecewise opacity transfer function
    otf = PiecewiseFunction()
    for v,o in zip(smpl,opac):
        otf.add_point(v,o)
    #
    return otf

def vol_rend(data_arr, dx, fig, otf, ctf):
    sc = mlab.pipeline.scalar_field(data_arr.T)
    sc.origin = (-100., -100., 0.)
    #dx = dset.attrs['dx']
    sc.spacing = (dx[0], dx[1], dx[2])

    sc.scalar_name = 'logf_xyz'

    vol = mlab.pipeline.volume(sc, vmin=6., vmax=13., figure=fig)
    # OTF
    vol._otf = otf
    vol._volume_property.set_scalar_opacity(otf)
    # CTF
    vol._volume_property.set_color(ctf)
    vol._ctf = ctf
    vol.update_ctf = True

    return vol

def test10():
    # --- loading data ---
    dfile = h5.File('../rho3d.h5', 'r')
    print(dfile.attrs.keys())
    print(dfile.attrs['mass ratio'])
    print(dfile['t=3072.'].attrs.keys())
    #
    dset = dfile['t=3072.']
    xyz = dset.attrs['grid']
    dx = dset.attrs['dx']
    rho_cgs = dset[:] * 6.176269145886166e+17
    rho_cgs = np.log10(rho_cgs)
    # rho_cgs = dset[:]
    print(rho_cgs.min(), rho_cgs.max())
    #
    rho_range = [6., 13.]

    centers = [8., 10., 11., 13.]
    opacs =   [0.2, 0.4, 0.6, 0.8]
    widths =  [0.2, 0.2, 0.2, 0.2]

    ''' -- '''

    fig = reset()

    ctf = get_ctf(crange=rho_range)
    otf = get_otf(centers, opacs, widths, orange=rho_range)
    vol = vol_rend(rho_cgs, dx, fig, otf, ctf)

    mlab.orientation_axes(figure=fig)
    mlab.show()
test10()