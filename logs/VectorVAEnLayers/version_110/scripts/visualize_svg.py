import glob
import os

import torch
import math
import svgpathtools
import pydiffvg
import argparse
import numpy as np
import torchvision.utils as vutils

render = pydiffvg.RenderFunction.apply
def make_tensor(x, grad=False):
    x = torch.tensor(x, dtype=torch.float32)
    x.requires_grad = grad
    return x

def hard_composite_(**kwargs):
    layers = kwargs['layers']
    n = len(layers)
    alpha = (1 - layers[n - 1][:, 3:4, :, :])
    rgb = layers[n - 1][:, :3] * layers[n - 1][:, 3:4, :, :]
    for i in reversed(range(n-1)):
        rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * alpha
        alpha = (1-layers[i][:, 3:4, :, :]) * alpha
    rgb = rgb + alpha
    return rgb

def hard_composite(**kwargs):
    layers = kwargs['layers']
    n = len(layers)
    alpha = (1 - layers[0][:, 3:4, :, :])
    rgb = layers[0][:, :3] * layers[0][:, 3:4, :, :]
    for i in range(1, n):
        rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * alpha
        alpha = (1-layers[i][:, 3:4, :, :]) * alpha
    rgb = rgb + alpha
    return rgb

def raster(all_points, color=[0, 0, 0, 1], verbose=False, white_background=True):
    assert len(color) == 4
    # print('1:', process.memory_info().rss*1e-6)
    render_size = 512
    paths = int(all_points.shape[0]/3)
    bs = 1#all_points.shape[0]
    outputs = []
    scaling =  torch.zeros([1,2])
    scaling[:, 0] = 512/24
    scaling[:, 1] = 512/24
    print(scaling)
    all_points = all_points * scaling
    num_ctrl_pts = torch.zeros(paths, dtype=torch.int32) + 2
    color = make_tensor(color)
    for k in range(bs):
        # Get point parameters from network
        shapes = []
        shape_groups = []
        points = all_points.cpu().contiguous()  # [self.sort_idx[k]]

        if verbose:
            np.random.seed(0)
            colors = np.random.rand(paths, 4)
            high = np.array((0.565, 0.392, 0.173, 1))
            low = np.array((0.094, 0.310, 0.635, 1))
            diff = (high - low) / (paths)
            colors[:, 3] = 1
            for i in range(paths):
                scale = diff * i
                color = low + scale
                color[3] = 1
                color = torch.tensor(color)
                num_ctrl_pts = torch.zeros(1, dtype=torch.int32) + 2
                if i * 3 + 4 > paths * 3:
                    curve_points = torch.stack([points[i * 3], points[i * 3 + 1], points[i * 3 + 2], points[0]])
                else:
                    curve_points = points[i * 3:i * 3 + 4]
                path = pydiffvg.Path(
                    num_control_points=num_ctrl_pts, points=curve_points,
                    is_closed=False, stroke_width=torch.tensor(4))
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([i]),
                    fill_color=None,
                    stroke_color=color)
                shapes.append(path)
                shape_groups.append(path_group)
            for i in range(paths * 3):
                scale = diff * (i // 3)
                color = low + scale
                color[3] = 1
                color = torch.tensor(color)
                if i % 3 == 0:
                    # color = torch.tensor(colors[i//3]) #green
                    shape = pydiffvg.Rect(p_min=points[i] - 8,
                                          p_max=points[i] + 8)
                    group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([paths + i]),
                                                fill_color=color)

                else:
                    # color = torch.tensor(colors[i//3]) #purple
                    shape = pydiffvg.Circle(radius=torch.tensor(8.0),
                                            center=points[i])
                    group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([paths + i]),
                                                fill_color=color)
                shapes.append(shape)
                shape_groups.append(group)

        else:

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                is_closed=True)

            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=color,
                stroke_color=color)
            shape_groups.append(path_group)
        scene_args = pydiffvg.RenderFunction.serialize_scene(render_size, render_size, shapes, shape_groups)
        out = render(render_size,  # width
                          render_size,  # height
                          2,  # num_samples_x
                          2,  # num_samples_y
                          102,  # seed
                          None,
                          *scene_args)
        out = out.permute(2, 0, 1).view(4, render_size, render_size)  # [:3]#.mean(0, keepdim=True)
        outputs.append(out)
    output = torch.stack(outputs).to(all_points.device)
    alpha = output[:, 3:4, :, :]

    # map to [-1, 1]
    if white_background:
        output_white_bg = output[:, :3, :, :] * alpha + (1 - alpha)
        output = torch.cat([output_white_bg, alpha], dim=1)
    del num_ctrl_pts, color
    return output


def from_svg_path(path_str, shape_to_canvas = torch.eye(3), force_close = False, verbose = True):
    colors = [[0, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], ]
    paths, attributes = svgpathtools.svg2paths(path_str)
    ret_paths = []
    paths = paths
    for path in paths:
        subpaths = path.continuous_subpaths()
        for idx, subpath in enumerate(subpaths):
            # print(subpath)
            if len(subpath)==0:
                continue
            if subpath.isclosed():
                if len(subpath) > 1 and isinstance(subpath[-1], svgpathtools.Line) and subpath[-1].length() < 1e-5:
                    subpath.remove(subpath[-1])
                    subpath[-1].end = subpath[0].start # Force closing the path
                    subpath.end = subpath[-1].end
                    assert(subpath.isclosed())
            else:
                beg = subpath[0].start
                end = subpath[-1].end
                if abs(end - beg) < 1e-5:
                    subpath[-1].end = beg # Force closing the path
                    subpath.end = subpath[-1].end
                    assert(subpath.isclosed())
                elif force_close:
                    subpath.append(svgpathtools.Line(end, beg))
                    subpath.end = subpath[-1].end
                    assert(subpath.isclosed())

            num_control_points = []
            points = []
            for i, e in enumerate(subpath):
                if i == 0:
                    points.append((e.start.real, e.start.imag))
                else:
                    # Must begin from the end of previous segment
                    assert(e.start.real == points[-1][0])
                    assert(e.start.imag == points[-1][1])
                if isinstance(e, svgpathtools.Line):
                    # num_control_points.append(0)
                    num_control_points.append(2)
                    points.append((e.start.real, e.start.imag))
                    points.append((e.start.real, e.start.imag))
                elif isinstance(e, svgpathtools.QuadraticBezier):
                    num_control_points.append(2)
                    points.append((e.control.real, e.control.imag))
                    points.append((e.control.real, e.control.imag))
                elif isinstance(e, svgpathtools.CubicBezier):
                    num_control_points.append(2)
                    points.append((e.control1.real, e.control1.imag))
                    points.append((e.control2.real, e.control2.imag))
                elif isinstance(e, svgpathtools.Arc):
                    # Convert to Cubic curves
                    # https://www.joecridge.me/content/pdf/bezier-arcs.pdf
                    start = e.theta * math.pi / 180.0
                    stop = (e.theta + e.delta) * math.pi / 180.0

                    sign = 1.0
                    if stop < start:
                        sign = -1.0

                    epsilon = 0.00001
                    debug = abs(e.delta) >= 90.0
                    while (sign * (stop - start) > epsilon):
                        arc_to_draw = stop - start
                        if arc_to_draw > 0.0:
                            arc_to_draw = min(arc_to_draw, 0.5 * math.pi)
                        else:
                            arc_to_draw = max(arc_to_draw, -0.5 * math.pi)
                        alpha = arc_to_draw / 2.0
                        cos_alpha = math.cos(alpha)
                        sin_alpha = math.sin(alpha)
                        cot_alpha = 1.0 / math.tan(alpha)
                        phi = start + alpha
                        cos_phi = math.cos(phi)
                        sin_phi = math.sin(phi)
                        lambda_ = (4.0 - cos_alpha) / 3.0
                        mu = sin_alpha + (cos_alpha - lambda_) * cot_alpha
                        last = sign * (stop - (start + arc_to_draw)) <= epsilon
                        num_control_points.append(2)
                        rx = e.radius.real
                        ry = e.radius.imag
                        cx = e.center.real
                        cy = e.center.imag
                        rot = e.phi * math.pi / 180.0
                        cos_rot = math.cos(rot)
                        sin_rot = math.sin(rot)
                        x = lambda_ * cos_phi + mu * sin_phi
                        y = lambda_ * sin_phi - mu * cos_phi
                        xx = x * cos_rot - y * sin_rot
                        yy = x * sin_rot + y * cos_rot
                        points.append((cx + rx * xx, cy + ry * yy))
                        x = lambda_ * cos_phi - mu * sin_phi
                        y = lambda_ * sin_phi + mu * cos_phi
                        xx = x * cos_rot - y * sin_rot
                        yy = x * sin_rot + y * cos_rot
                        points.append((cx + rx * xx, cy + ry * yy))
                        if not last:
                            points.append((cx + rx * math.cos(rot + start + arc_to_draw),
                                           cy + ry * math.sin(rot + start + arc_to_draw)))
                        start += arc_to_draw
                        first = False
                if i != len(subpath) - 1:
                    points.append((e.end.real, e.end.imag))
                else:
                    if subpath.isclosed():
                        # Must end at the beginning of first segment
                        assert(e.end.real == points[0][0])
                        assert(e.end.imag == points[0][1])
                    else:
                        points.append((e.end.real, e.end.imag))
            points = torch.tensor(points)
            points = torch.cat((points, torch.ones([points.shape[0], 1])), dim = 1) @ torch.transpose(shape_to_canvas, 0, 1)
            points = points / points[:, 2:3]
            points = points[:, :2].contiguous()
            if verbose:
                ret_paths.append(raster(points, verbose=True))
            else:
                # if i==0:
                #     continue
                if i>len(colors)-1:
                    i = -1
                print(i)
                ret_paths.append(raster(points, colors[idx], verbose=False))
    return ret_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", help="source SVG path")
    args = parser.parse_args()
    svg_folder = os.path.join(args.svg)
    svgs = glob.glob(svg_folder+'/*.svg')
    renders = []
    for file in range(len(svgs)):
        name = svg_folder+f'/{file}.svg'
        print(name)
        layers = from_svg_path(name, verbose = False)
        composite = hard_composite_(layers=layers)
        renders.append(composite)
    render = torch.cat(renders, dim=0)
    vutils.save_image(render.cpu().data,
                      svg_folder+
                      f"/img.png",
                      normalize=False,
                      nrow=10)