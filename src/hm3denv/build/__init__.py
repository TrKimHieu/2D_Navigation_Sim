"""Dataset building tools (``pip install hm3denv[build]``).

Stages, each a module with explicit parameters (no hard-coded paths):

    slice       GLB -> per-storey 2 cm obstacle/floor/roof maps   (workspace stages/slice/)
    review      web tool to fix up axes and storeys by hand          (review.json)
    gridify     storey maps -> robot grids                           (dataset grids/)
    tasks_grid  grid tasks + labels, checked on the mesh             (dataset tasks/, verify/)
    vectorize   storey maps -> SVG maps per height class             (dataset maps/)
    tasks_svg   SVG tasks per robot, checked on the mesh             (dataset tasks/, verify/)
    verify      3D mesh checks shared by the samplers + re-check
    preview     PNG/SVG previews and contact sheets
    urdf        real robot footprints from manufacturer URDFs
    pipeline    config file -> dataset, with stage stamps            (``hm3d build``)
"""
