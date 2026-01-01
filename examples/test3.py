import gdsfactory as gf
import numpy as np

# -----------------------------------------------------------------------------
# Layer Mapping
# -----------------------------------------------------------------------------
LAYER_TOP = (134, 0)   # TM2: Top Winding
LAYER_BOT = (126, 0)    # TM1: Bottom Winding
LAYER_RING = (67, 0)  # Ground Ring

@gf.cell
def tf_octa_c(
    di: float = 50.0,
    do: float = 50.0,
    dis: float = 10.0,
    wi: float = 5.0,
    wic: float = 0.0,
    fi: int = 1,
    wo: float = 5.0,
    woc: float = 0.0,
    fo: int = 1,
    fs: float = 6.0,
    ro: float = 10.0,
    ri: float = 10.0,
    rs: float = 10.0,
    rw: float = 10.0,
) -> gf.Component:
    """
    Octagon Transformer Component (tf_octa_c).
    
    Args:
        di: Diameter of input (lower/bot) winding (trace center to center).
        do: Diameter of output (upper/top) winding (trace center to center).
        dis: Displacement (offset between winding centers).
        wi: Trace width of lower winding.
        wic: Trace width of lower center tap.
        fi: Feed type lower (0=no, 1=fwd, 2=rev, 3=both).
        wo: Trace width of upper winding.
        woc: Trace width of upper center tap.
        fo: Feed type upper (0=no, 1=fwd, 2=rev, 3=both).
        fs: Feedline spacing (gap between inner sides of feed lines).
        ro: Ring spacing on upper winding side.
        ri: Ring spacing on lower winding side.
        rs: Ring spacing at side.
        rw: Ring width.
    """
    c = gf.Component("tf_octa_c")

    # -------------------------------------------------
    # 1. Variable Calculation
    # -------------------------------------------------
    wic_int = wic if wic > 0.1 else wi
    woc_int = woc if woc > 0.1 else wo
    fo_int = int(round(fo))
    fi_int = int(round(fi))

    # Bounding Geometry
    tf_y = max(do, di) / 2.0 + rs
    
    # Calculate Port X limits [cite: 1]
    right_edge_top = (dis / 2.0) + (do / 2.0)
    right_edge_bot = (-dis / 2.0) + (di / 2.0)
    port_xr = max(right_edge_top, right_edge_bot) + ro
    
    left_edge_top = (dis / 2.0) - (do / 2.0)
    left_edge_bot = (-dis / 2.0) - (di / 2.0)
    port_xl = min(left_edge_top, left_edge_bot) - ri

    # -------------------------------------------------
    # 2. Winding Generation Helper
    # -------------------------------------------------
    def create_octa_winding(diameter, width, spacing, layer, center_x, center_y, rotation_deg):
        """Generates the octagonal winding path with feed gap."""
        r = diameter / 2.0
        # Start at top of feed gap, go CCW to bottom of feed gap
        # Feed gap is vertical of height 'spacing'
        # Simplified octagon vertices generation
        x_face = r * np.cos(np.radians(22.5))
        
        path_pts = []
        path_pts.append((x_face, spacing/2.0)) # Start
        
        # Intermediate vertices (22.5 to 337.5)
        for ang in np.arange(22.5, 360, 45):
            rad = np.radians(ang)
            path_pts.append((r * np.cos(rad), r * np.sin(rad)))
            
        path_pts.append((x_face, -spacing/2.0)) # End
        
        p = gf.Path(path_pts)
        ref = c << p.extrude(width=width, layer=layer)
        ref.rotate(rotation_deg)
        ref.move((center_x, center_y))
        return ref

    # Draw Windings
    # Top (Gap Right)
    create_octa_winding(do, wo, fs, LAYER_TOP, dis/2.0, 0, 0)
    # Bot (Gap Left)
    create_octa_winding(di, wi, fs, LAYER_BOT, -dis/2.0, 0, 180)

    # -------------------------------------------------
    # 3. Main Ports (ip, in, op, on)
    # -------------------------------------------------
    # Unique Layers: 201-204
    
    # IP (Input Positive) - Bot [cite: 4]
    c.add_port(
        name="ip", center=(port_xl, fs/2.0), width=wi, orientation=180, layer=(201, 0)
    )
    # IN (Input Negative) - Bot [cite: 4]
    c.add_port(
        name="in", center=(port_xl, -fs/2.0 - wi), width=wi, orientation=180, layer=(202, 0)
    )
    # OP (Output Positive) - Top [cite: 2]
    c.add_port(
        name="op", center=(port_xr, fs/2.0), width=wo, orientation=0, layer=(203, 0)
    )
    # ON (Output Negative) - Top [cite: 2]
    c.add_port(
        name="on", center=(port_xr, -fs/2.0 - wo), width=wo, orientation=0, layer=(204, 0)
    )

    # -------------------------------------------------
    # 4. Center Taps & Ports (ici, ico, oci, oco)
    # -------------------------------------------------
    # Unique Layers: 205-208
    
    # --- ICI (Input Center, Input Side) --- [cite: 3]
    # Default: Back of Bot winding (Right side of Bot trace)
    pos_ici_x = -dis/2.0 + di/2.0 - wi/2.0
    width_ici = 0.25 # Default pin width
    
    if fi_int & 2: # Feed Type: Rev (extends to Left/Input side)
        width_ici = wic_int
        pos_ici_x = port_xl
        # Draw Wire
        p = gf.Path([(-dis/2.0 + di/2.0, 0), (port_xl, 0)])
        c << p.extrude(width=wic_int, layer=LAYER_BOT)
        
    c.add_port(name="ici", center=(pos_ici_x, 0), width=width_ici, orientation=180, layer=(205, 0))

    # --- ICO (Input Center, Output Side) --- [cite: 4]
    # Default: Back of Bot winding (Right side of Bot trace)
    # Note: Skill calc for default tmp includes +wi/2, likely referencing outer edge or specific pin alignment
    pos_ico_x = -dis/2.0 + di/2.0 + wi/2.0 
    width_ico = 0.25
    
    if fi_int & 1: # Feed Type: Fwd (extends to Right/Output side)
        width_ico = wic_int
        pos_ico_x = port_xr
        # Draw Wire
        p = gf.Path([(port_xr, 0), (-dis/2.0 + di/2.0, 0)])
        c << p.extrude(width=wic_int, layer=LAYER_BOT)

    c.add_port(name="ico", center=(pos_ico_x, 0), width=width_ico, orientation=0, layer=(206, 0))

    # --- OCI (Output Center, Input Side) --- [cite: 3]
    # Default: Back of Top winding (Left side of Top trace)
    pos_oci_x = dis/2.0 - do/2.0 - wo/2.0
    width_oci = 0.25
    
    if fo_int & 1: # Feed Type: Fwd (extends to Left/Input side)
        width_oci = woc_int
        pos_oci_x = port_xl
        # Draw Wire
        p = gf.Path([(port_xl, 0), (dis/2.0 - do/2.0, 0)])
        c << p.extrude(width=woc_int, layer=LAYER_TOP)

    c.add_port(name="oci", center=(pos_oci_x, 0), width=width_oci, orientation=180, layer=(207, 0))

    # --- OCO (Output Center, Output Side) --- [cite: 5]
    # Default: Back of Top winding (Left side of Top trace)
    pos_oco_x = dis/2.0 - do/2.0 + wo/2.0
    width_oco = 0.25
    
    if fo_int & 2: # Feed Type: Rev (extends to Right/Output side)
        width_oco = woc_int
        pos_oco_x = port_xr
        # Draw Wire
        p = gf.Path([(dis/2.0 - do/2.0, 0), (port_xr, 0)])
        c << p.extrude(width=woc_int, layer=LAYER_TOP)

    c.add_port(name="oco", center=(pos_oco_x, 0), width=width_oco, orientation=0, layer=(208, 0))

    # -------------------------------------------------
    # 5. Ground Ring
    # -------------------------------------------------
    ring_pts = [
        (0, tf_y), (port_xl, tf_y), (port_xl, -tf_y), 
        (port_xr, -tf_y), (port_xr, tf_y), (0, tf_y)
    ]
    p_ring = gf.Path(ring_pts)
    c << p_ring.extrude(width=rw, layer=LAYER_RING)
    
    return c

if __name__ == "__main__":
    c = tf_octa_c(
        di=50.0,
        do=50.0,
        dis=10.0,
        wi=7.0,
        wic=0.0,
        fi=1,
        wo=7.0,
        woc=0.0,
        fo=1,
        fs=8.0,
        ro=10.0,
        ri=10.0,
        rs=10.0,
        rw=10.0,
    )
    c.show()