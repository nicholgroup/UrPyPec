import ezdxf

import polyCreate

for dxf in ['saw.dxf', '2D_v2.dxf', 'spiral_nice.dxf', 'straps.dxf', 'grating.dxf',
            'spiral_nice_simple', 'spiral_bad.dxf', '2D.dxf', 'P7.dxf',   'dot.dxf', 'mm.dxf']:
    dataDXF = ezdxf.readfile(dxf)
    polyCreate.polyfromdxf(dxinfo=dataDXF)

# dataDXF = ezdxf.readfile('2D.dxf')
# polyCreate.polyfromdxf(dxinfo=dataDXF)
# # dot, p7, and mm do not work. 2dv2 has a small in polygon creation issue.
