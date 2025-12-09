#!/usr/bin/env python3
"""
Minimal Dash app around examples/extrude_and_cap.py.

- Loads an STL (default: master_combined_open.stl)
- Exposes controls for end selection, extrusion length, scale, and segments
- Writes the updated STL to --output (default: out.stl)
- Renders the resulting surface via Plotly Mesh3d
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

import extrude_and_cap as extrude_mod


def run_pipeline(base_mesh: trimesh.Trimesh,
                 end_name: str,
                 length: float,
                 scale: float,
                 segments: int,
                 output_path: Path):
    """Extrude, cap, and write the STL; return the finished mesh and stats."""
    extruded, meta = extrude_mod.extrude_narrow_end(
        base_mesh,
        end_name=end_name,
        length=length,
        scale_target=scale,
        segments=segments,
    )
    capped, capped_info = extrude_mod.cap_open_ends(extruded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capped.export(output_path)
    return capped, meta, capped_info


def axis_guides(mesh: trimesh.Trimesh, go):
    """Return colored axis guide traces anchored at the mesh center."""
    bounds = mesh.bounds
    origin = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    guide_len = 0.25 * float(max(extent.max(), 1.0))
    axes = [
        ("X", np.array([guide_len, 0, 0]), "#d62728"),
        ("Y", np.array([0, guide_len, 0]), "#2ca02c"),
        ("Z", np.array([0, 0, guide_len]), "#1f77b4"),
    ]
    traces = []
    for label, vec, color in axes:
        end = origin + vec
        traces.append(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines+text",
                text=["", label],
                textposition="top center",
                line=dict(color=color, width=6),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return traces


def make_figure(mesh: trimesh.Trimesh, go):
    """Convert a trimesh mesh to a Plotly Mesh3d figure."""
    verts = mesh.vertices
    faces = mesh.faces
    trace = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="#1f77b4",
        opacity=0.7,
        flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.6, specular=0.35, roughness=0.9),
    )
    fig = go.Figure(data=[trace, *axis_guides(mesh, go)])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", visible=True, showgrid=True, zeroline=True),
            yaxis=dict(title="Y", visible=True, showgrid=True, zeroline=True),
            zaxis=dict(title="Z", visible=True, showgrid=True, zeroline=True),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title="Extruded and capped STL",
    )
    return fig


def create_app(base_mesh: trimesh.Trimesh, output_path: Path, defaults: dict):
    try:
        from dash import Dash, Input, Output, dcc, html  # type: ignore
        import plotly.graph_objects as go
    except Exception:
        print("Dash/Plotly not installed. Install with: pip install dash plotly")
        raise

    app = Dash(__name__)

    end_options = [
        {"label": "YZ_maxX", "value": "YZ_maxX"},
        {"label": "XZ_minY", "value": "XZ_minY"},
        {"label": "XZ_maxY", "value": "XZ_maxY"},
        {"label": "XY_maxZ", "value": "XY_maxZ"},
    ]

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Label("End outlet"),
                    dcc.Dropdown(
                        id="end-dropdown",
                        options=end_options,
                        value=defaults["end"],
                        clearable=False,
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Label("Length"),
                    dcc.Input(
                        id="length-input",
                        type="number",
                        value=defaults["length"],
                        step=0.5,
                        min=0,
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Label("Scale"),
                    dcc.Slider(
                        id="scale-slider",
                        min=0.01,
                        max=1.0,
                        step=0.01,
                        value=defaults["scale"],
                        marks={0.01: "0.01", 1.0: "1.0"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Label("Segments"),
                    dcc.Input(
                        id="segments-input",
                        type="number",
                        value=defaults["segments"],
                        step=1,
                        min=1,
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    dcc.Loading(dcc.Graph(id="mesh-figure", style={"height": "70vh"})),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Pre(id="status-text", style={"whiteSpace": "pre-wrap"}),
        ],
        style={"maxWidth": "960px", "margin": "0 auto", "padding": "12px"},
    )

    @app.callback(
        Output("mesh-figure", "figure"),
        Output("status-text", "children"),
        Input("end-dropdown", "value"),
        Input("length-input", "value"),
        Input("scale-slider", "value"),
        Input("segments-input", "value"),
    )
    def update_mesh(end_name, length, scale, segments):
        end_val = end_name or defaults["end"]
        length_val = defaults["length"] if length is None else float(length)
        scale_val = defaults["scale"] if scale is None else float(scale)
        seg_val = defaults["segments"] if segments is None else int(segments)
        seg_val = max(1, seg_val)

        try:
            mesh, meta, caps = run_pipeline(
                base_mesh,
                end_val,
                length_val,
                scale_val,
                seg_val,
                output_path,
            )
        except Exception as exc:  # pragma: no cover - interactive path
            fig_err = go.Figure()
            fig_err.update_layout(title=f"Failed to build geometry: {exc}")
            return fig_err, f"Error while updating geometry: {exc}"

        fig = make_figure(mesh, go)
        status_lines = [
            f"Saved to: {output_path}",
            f"End: {end_val} | length: {length_val:.2f} | scale: {scale_val:.2f} | segments: {seg_val}",
            f"Vertices: {len(mesh.vertices):,} | Faces: {len(mesh.faces):,}",
            f"Extrusion meta: {meta}",
            f"Capped faces per end: {caps}",
        ]
        return fig, "\n".join(status_lines)

    return app


def main():
    parser = argparse.ArgumentParser(description="Dash viewer for extrude_and_cap.py output")
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--input",
        type=str,
        default=str(repo_root / "master_combined_open.stl"),
        help="Input STL to extrude",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(repo_root / "out.stl"),
        help="Destination STL written on each update",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input STL not found: {input_path}")

    base_mesh = trimesh.load_mesh(input_path)
    defaults = {"end": "YZ_maxX", "length": 14.0, "scale": 0.75, "segments": 10}

    app = create_app(base_mesh, output_path, defaults)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
