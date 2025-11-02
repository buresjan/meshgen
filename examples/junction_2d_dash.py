#!/usr/bin/env python3
"""
Dash slice viewer for labeled voxels from the junction_2d example.

This script generates the junction_2d geometry via the .geo → STL → voxels
route (using the same parameters as examples/junction_2d_visualize.py),
converts the boolean occupancy to simulation labels (0/1/2/3/4/5 and optional
11..16) using the same pipeline used for text export, and serves a Dash app
to explore 2D slices through the 3D labeled volume.

Usage:
  # ensure dash+plotly are installed
  #   conda activate meshgen
  #   pip install dash plotly
  python examples/junction_2d_dash.py

Notes:
- Visualization is slice-based for performance and clarity of label categories.
- Colors are discrete by label; a legend is shown on the right.
"""

import os
import sys
import argparse
import numpy as np

# Try to import the package; add repo root if running without installation
try:
    from meshgen.geometry import Geometry
    from meshgen.voxels import prepare_voxel_mesh_txt
except Exception:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from meshgen.geometry import Geometry
    from meshgen.voxels import prepare_voxel_mesh_txt


def compute_labels(resolution: int = 6, expected_faces=None):
    if expected_faces is None:
        expected_faces = {"W", "E", "N", "S"}
    geom = Geometry(
        name="junction_2d",
        resolution=resolution,
        split=None,
        num_processes=1,
        output_dir="output",
        expected_in_outs=expected_faces,
        lower_angle=15,
        upper_angle=15,
        upper_flare=0.001,
        lower_flare=0.001,
    )
    geom.generate_voxel_mesh()
    occ = geom.get_voxel_mesh()
    # Produce labels both with and without face tags for comparison
    labels_faces = prepare_voxel_mesh_txt(occ, expected_in_outs=expected_faces, num_type="int", label_faces=True)
    labels_nofaces = prepare_voxel_mesh_txt(occ, expected_in_outs=expected_faces, num_type="int", label_faces=False)
    return occ, labels_faces, labels_nofaces


def to_rgb_image(slice2d: np.ndarray, visible_labels: set):
    """Map an integer-labeled 2D slice to an RGB uint8 image using a discrete palette.

    Labels and colors:
      0 empty      → (0, 0, 0)
      1 fluid      → (31, 119, 180)
      2 wall       → (127, 127, 127)
      3 near-wall  → (44, 160, 44)
      4 near^2     → (255, 221, 87)
      5 near^3     → (255, 127, 14)
     11 N-face     → (214, 39, 40)
     12 E-face     → (227, 119, 194)
     13 S-face     → (148, 103, 189)
     14 W-face     → (140, 86, 75)
     15 B-face     → (23, 190, 207)
     16 F-face     → (255, 152, 150)
    """
    palette = {
        0: (0, 0, 0),
        1: (31, 119, 180),
        2: (127, 127, 127),
        3: (44, 160, 44),
        4: (255, 221, 87),
        5: (255, 127, 14),
        11: (214, 39, 40),
        12: (227, 119, 194),
        13: (148, 103, 189),
        14: (140, 86, 75),
        15: (23, 190, 207),
        16: (255, 152, 150),
    }

    h, w = slice2d.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Default all to background (0)
    img[:, :, :] = palette[0]
    labels_present = np.unique(slice2d)
    for lbl in labels_present:
        if lbl not in palette:
            continue
        if lbl in visible_labels:
            mask = slice2d == lbl
            img[mask] = palette[lbl]
    return img


def main():
    parser = argparse.ArgumentParser(description="Dash slice viewer for junction_2d labeled voxels")
    parser.add_argument("--resolution", type=int, default=6, help="Resolution scale; longest axis ~ 128*res")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind")
    parser.add_argument(
        "--faces",
        type=str,
        default="W,E,N,S",
        help="Comma-separated domain faces to tag (subset of N,E,S,W,B,F)",
    )
    args = parser.parse_args()

    # Compute labels once up-front
    faces = {f.strip().upper() for f in args.faces.split(',') if f.strip()}
    faces &= {"N", "E", "S", "W", "B", "F"}
    if not faces:
        faces = {"W", "E", "N", "S"}
    occ, labels_faces, labels_nofaces = compute_labels(args.resolution, expected_faces=faces)
    # Default view uses face tags
    labels = labels_faces
    sx, sy, sz = labels.shape

    # Late import Dash/Plotly to keep optional
    try:
        from dash import Dash, dcc, html, Input, Output
        import plotly.graph_objects as go
    except Exception as e:
        print("Dash/Plotly not installed. Install with: pip install dash plotly")
        raise

    app = Dash(__name__)

    # Shared label mapping and order for histograms and hover
    label_names_map = {
        0: "outside (0)",
        1: "fluid (1)",
        2: "wall (2)",
        3: "near-wall (3)",
        4: "near^2 (4)",
        5: "near^3 (5)",
        11: "N-face (11)",
        12: "E-face (12)",
        13: "S-face (13)",
        14: "W-face (14)",
        15: "B-face (15)",
        16: "F-face (16)",
    }
    label_order = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16]

    def counts_for(arr2d_or_3d):
        vals, cnts = np.unique(arr2d_or_3d, return_counts=True)
        d = {int(v): int(c) for v, c in zip(vals, cnts)}
        return [d.get(v, 0) for v in label_order]

    # Simple HTML table builder for absolute counts
    def build_table(counts, title):
        header = html.Tr([html.Th("Label"), html.Th("Name"), html.Th("Count", style={"textAlign": "right"})])
        rows = []
        total = 0
        for v, c in zip(label_order, counts):
            total += int(c)
            rows.append(html.Tr([
                html.Td(str(v)),
                html.Td(label_names_map.get(v, f"label {v}")),
                html.Td(f"{int(c):,}", style={"textAlign": "right"}),
            ]))
        rows.append(html.Tr([
            html.Td("", style={"borderTop": "1px solid #ccc"}),
            html.Td("Total", style={"fontWeight": "bold", "borderTop": "1px solid #ccc"}),
            html.Td(f"{total:,}", style={"textAlign": "right", "fontWeight": "bold", "borderTop": "1px solid #ccc"}),
        ]))
        table = html.Table([html.Thead(header), html.Tbody(rows)], style={"width": "100%", "borderCollapse": "collapse"})
        return html.Div([html.H5(title), table])

    # Precompute full-volume counts and tables (faces on/off)
    full_counts_faces = counts_for(labels_faces)
    full_counts_nofaces = counts_for(labels_nofaces)
    full_table_faces = build_table(full_counts_faces, "Volume Label Counts (faces on)")
    full_table_nofaces = build_table(full_counts_nofaces, "Volume Label Counts (faces off)")

    # Default visible labels: all known categories
    all_labels = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16]
    default_visible = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16]  # hide 0 by default

    def legend_swatches():
        items = [
            (1, "fluid (1)", "rgb(31,119,180)"),
            (2, "wall (2)", "rgb(127,127,127)"),
            (3, "near-wall (3)", "rgb(44,160,44)"),
            (4, "near^2 (4)", "rgb(255,221,87)"),
            (5, "near^3 (5)", "rgb(255,127,14)"),
            (11, "N-face (11)", "rgb(214,39,40)"),
            (12, "E-face (12)", "rgb(227,119,194)"),
            (13, "S-face (13)", "rgb(148,103,189)"),
            (14, "W-face (14)", "rgb(140,86,75)"),
            (15, "B-face (15)", "rgb(23,190,207)"),
            (16, "F-face (16)", "rgb(255,152,150)"),
            (0, "outside (0)", "rgb(0,0,0)"),
        ]
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(style={"width": "14px", "height": "14px", "background": c, "display": "inline-block", "marginRight": "6px", "border": "1px solid #333"}),
                        html.Span(lbl),
                    ],
                    style={"marginBottom": "6px"},
                )
                for _, lbl, c in items
            ]
        )

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H4("junction_2d labeled voxels — slice viewer"),
                    html.Div(f"volume shape: {labels.shape}")
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Axis"),
                            dcc.Dropdown(
                                id="axis",
                                options=[
                                    {"label": "X", "value": "x"},
                                    {"label": "Y", "value": "y"},
                                    {"label": "Z", "value": "z"},
                                ],
                                value="z",
                                clearable=False,
                                style={"width": "120px"},
                            ),
                            html.Label("Slice index"),
                            dcc.Slider(id="slice", min=0, max=sz - 1, step=1, value=sz // 2),
                            html.Label("Visible labels"),
                            dcc.Checklist(
                                id="visible",
                                options=[{"label": str(v), "value": v} for v in all_labels],
                                value=default_visible,
                                inline=True,
                            ),
                            html.Label("Show face tags"),
                            dcc.Checklist(
                                id="show-faces",
                                options=[{"label": "face tags", "value": "on"}],
                                value=["on"],
                                inline=True,
                            ),
                        ],
                        style={"flex": "1", "paddingRight": "12px"},
                    ),
                    html.Div(legend_swatches(), style={"width": "220px", "borderLeft": "1px solid #ddd", "paddingLeft": "12px"}),
                ],
                style={"display": "flex", "alignItems": "flex-start"},
            ),
            dcc.Graph(id="slice-graph", style={"height": "70vh"}),
            html.Div(
                [
                    html.Div(id="table-full", style={"flex": "1", "paddingRight": "8px", "overflow": "auto"}),
                    html.Div(id="table-slice", style={"flex": "1", "paddingLeft": "8px", "overflow": "auto"}),
                ],
                style={"display": "flex", "width": "100%", "marginTop": "12px"},
            ),
        ],
        style={"padding": "12px"},
    )

    @app.callback(
        Output("slice", "max"),
        Output("slice", "value"),
        Input("axis", "value"),
    )
    def update_slider(axis):
        if axis == "x":
            return sx - 1, sx // 2
        if axis == "y":
            return sy - 1, sy // 2
        return sz - 1, sz // 2

    @app.callback(
        Output("slice-graph", "figure"),
        Output("table-slice", "children"),
        Output("table-full", "children"),
        Input("axis", "value"),
        Input("slice", "value"),
        Input("visible", "value"),
        Input("show-faces", "value"),
    )
    def render_slice(axis, idx, visible_vals, show_faces_vals):
        use_faces = bool(show_faces_vals and ("on" in show_faces_vals))
        arr = labels_faces if use_faces else labels_nofaces
        vis = set(int(v) for v in (visible_vals or []))
        if axis == "x":
            sl = arr[idx, :, :].T  # transpose to render with x horizontal
            sl = sl[::-1, :]       # flip Z so z=0 at bottom
            x_title, y_title = "Y", "Z"
        elif axis == "y":
            sl = arr[:, idx, :].T
            sl = sl[::-1, :]       # flip Z so z=0 at bottom
            x_title, y_title = "X", "Z"
        else:
            sl = arr[:, :, idx].T
            x_title, y_title = "X", "Y"

        img = to_rgb_image(sl, vis)

        # Human-readable names for hover using shared map
        def lname(v: int) -> str:
            return label_names_map.get(int(v), f"label {int(v)}")
        hoverdata = np.vectorize(lname)(sl)

        fig = go.Figure()
        # Attach per-pixel hover via customdata + hovertemplate
        fig.add_trace(
            go.Image(
                z=img,
                customdata=hoverdata,
                hovertemplate="(%{x}, %{y}) %{customdata}<extra></extra>",
            )
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=24, b=0),
            xaxis=dict(scaleanchor="y", title=x_title),
            yaxis=dict(title=y_title),
        )
        # Slice counts table
        slice_counts = counts_for(sl)
        slice_table = build_table(slice_counts, "Slice Label Counts")
        full_table = full_table_faces if use_faces else full_table_nofaces
        return fig, slice_table, full_table

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
