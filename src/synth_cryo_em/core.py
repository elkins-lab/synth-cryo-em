import gemmi
import numpy as np
import mrcfile
from scipy.ndimage import gaussian_filter

def generate_density_map(input_path, resolution, grid_spacing=None):
    """
    Generate a density map from an atomic model file (PDB, mmCIF, BCIF) using gemmi.
    """
    st = gemmi.read_structure(input_path)
    # st.setup_entities() # Not strictly necessary for density
    
    # If grid_spacing is not provided, use a rule of thumb (resolution / 3 or 4)
    if grid_spacing is None:
        grid_spacing = resolution / 3.0
        
    # Get all atomic positions
    positions = []
    for model in st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    positions.append(atom.pos.tolist())
    
    if not positions:
        raise ValueError("No atoms found in structure")

    positions = np.array(positions)
    margin = resolution * 2.0 # Increased margin
    min_pos = positions.min(axis=0) - margin
    max_pos = positions.max(axis=0) + margin
    size = max_pos - min_pos
    
    size = max_pos - min_pos
    
    st_shifted = st.clone()
    cell = gemmi.UnitCell(size[0], size[1], size[2], 90, 90, 90)
    st_shifted.cell = cell
    
    for model in st_shifted:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.pos.x -= min_pos[0]
                    atom.pos.y -= min_pos[1]
                    atom.pos.z -= min_pos[2]

    # Map the atoms to the grid using DensityCalculatorE
    calc = gemmi.DensityCalculatorE()
    calc.d_min = resolution
    # set_grid_cell_and_spacegroup will use the cell we set on st_shifted
    calc.set_grid_cell_and_spacegroup(st_shifted)
    # If we want a specific spacing, we can set it
    if grid_spacing:
        # DensityCalculator might not allow direct spacing setting easily
        # but it calculates it from d_min if not specified.
        # Let's see if we can set rate
        calc.rate = resolution / grid_spacing
        
    calc.initialize_grid()
    if len(st_shifted) > 0:
        calc.put_model_density_on_grid(st_shifted[0])
    
    return calc.grid, min_pos

def add_gaussian_noise(data, snr):
    """
    Add Gaussian noise to the data based on desired SNR.
    """
    signal_power = np.mean(data**2)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise

def apply_ctf(data, voxel_size, defoc=2.0, cs=2.7, voltage=300, amplitude_contrast=0.1, b_factor=0.0):
    """
    Apply a simple Contrast Transfer Function (CTF) to the 3D data.
    defoc: defocus in micrometers
    cs: spherical aberration in mm
    voltage: acceleration voltage in kV
    b_factor: envelope function B-factor
    """
    # Constants
    wl = 12.26 / np.sqrt(voltage * 1000 + 0.9784 * voltage**2) # wavelength in Angstroms
    cs_a = cs * 1e7 # cs in Angstroms
    defoc_a = defoc * 10000 # defocus in Angstroms
    
    nz, ny, nx = data.shape
    # Frequencies
    kz = np.fft.fftfreq(nz, d=voxel_size[2])
    ky = np.fft.fftfreq(ny, d=voxel_size[1])
    kx = np.fft.fftfreq(nx, d=voxel_size[0])
    
    Kz, Ky, Kx = np.meshgrid(kz, ky, kx, indexing='ij')
    k2 = Kz**2 + Ky**2 + Kx**2
    
    # Phase shift
    chi = np.pi * wl * k2 * (defoc_a - 0.5 * wl**2 * k2 * cs_a)
    
    # CTF
    ctf = - (np.sqrt(1 - amplitude_contrast**2) * np.sin(chi) + amplitude_contrast * np.cos(chi))
    
    # Envelope function
    if b_factor > 0:
        envelope = np.exp(-b_factor * k2 / 4.0)
        ctf *= envelope
        
    # Apply in Fourier domain
    data_f = np.fft.fftn(data)
    data_f *= ctf
    return np.real(np.fft.ifftn(data_f))

def save_mrc(data, output_path, origin=(0,0,0), spacing=(1,1,1)):
    """
    Save numpy array to MRC file.
    """
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = spacing
        # mrcfile uses x, y, z for origin
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]
        mrc.update_header_from_data()
