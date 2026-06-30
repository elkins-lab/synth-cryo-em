import gemmi
import numpy as np
import mrcfile
from scipy.ndimage import gaussian_filter

def generate_density_map(input_path: str, resolution: float, grid_spacing: float = None, use_bfactors: bool = False, margin: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a density map from an atomic model file (PDB, mmCIF, BCIF) using gemmi.

    Args:
        input_path (str): Path to the input structure file.
        resolution (float): Target resolution in Angstroms.
        grid_spacing (float, optional): Grid spacing (pixel size) in Angstroms. Defaults to resolution / 3.0.
        use_bfactors (bool, optional): If True, use atomic B-factors for local resolution. Defaults to False.
        margin (float, optional): Margin around the structure in Angstroms. Defaults to resolution * 2.0.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The 3D density grid.
            - numpy.ndarray: The origin coordinate of the map.
    """
    st = gemmi.read_structure(input_path)
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
    if margin is None:
        margin = resolution * 2.0
    
    min_pos = positions.min(axis=0) - margin
    max_pos = positions.max(axis=0) + margin
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
    
    # Calculate sampling rate to match grid_spacing
    # spacing = d_min / (2 * rate)  => rate = d_min / (2 * spacing)
    calc.rate = resolution / (2.0 * grid_spacing)
    
    # Initialize grid
    calc.set_grid_cell_and_spacegroup(st_shifted)
    calc.initialize_grid()
    
    # If use_bfactors is True, we use the atomic B-factors.
    # Gemmi's DensityCalculatorE uses atomic B-factors by default 
    # when calling put_model_density_on_grid.
    # However, we can add a constant "base" blur to represent resolution.
    # resolution (d) relates to B-factor roughly by B = 8 * pi^2 * (d/2)^2 = 2 * pi^2 * d^2
    # But gemmi also uses d_min as a cutoff.
    
    if not use_bfactors:
        # If not using B-factors, we set them all to 0 and use a global blur
        # equivalent to the target resolution.
        for model in st_shifted:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom.b_iso = 0.0
        # Set a global blur to match the target resolution
        # A common heuristic is B = 8 * res^2 for synthetic maps
        calc.blur = 8.0 * resolution**2
    
    calc.initialize_grid()
    if len(st_shifted) > 0:
        calc.put_model_density_on_grid(st_shifted[0])
    
    return calc.grid, min_pos

def add_gaussian_noise(data: np.ndarray, snr: float) -> np.ndarray:
    """
    Add Gaussian noise to the data based on desired SNR.

    Args:
        data (numpy.ndarray): The input 3D density map.
        snr (float): Target Signal-to-Noise Ratio.

    Returns:
        numpy.ndarray: The noisy density map.
    """
    signal_power = np.mean(data**2)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise

def apply_ctf(data: np.ndarray, voxel_size: tuple[float, float, float], defoc: float = 2.0, cs: float = 2.7, voltage: float = 300.0, amplitude_contrast: float = 0.1, b_factor: float = 0.0) -> np.ndarray:
    """
    Apply a simple Contrast Transfer Function (CTF) to the 3D data.

    Args:
        data (numpy.ndarray): The input 3D density map.
        voxel_size (tuple): Voxel size in Angstroms (x, y, z).
        defoc (float, optional): Defocus in micrometers. Defaults to 2.0.
        cs (float, optional): Spherical aberration in mm. Defaults to 2.7.
        voltage (float, optional): Acceleration voltage in kV. Defaults to 300.
        amplitude_contrast (float, optional): Amplitude contrast fraction. Defaults to 0.1.
        b_factor (float, optional): Envelope function B-factor. Defaults to 0.0.

    Returns:
        numpy.ndarray: The CTF-corrupted 3D density map.
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

def compute_fsc(data1: np.ndarray, data2: np.ndarray, voxel_size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fourier Shell Correlation (FSC) between two 3D maps.

    Args:
        data1 (numpy.ndarray): The first 3D density map.
        data2 (numpy.ndarray): The second 3D density map.
        voxel_size (tuple): Voxel size in Angstroms (x, y, z).

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The spatial frequencies (1/Angstroms).
            - numpy.ndarray: The computed FSC values.
    """
    assert data1.shape == data2.shape
    
    # Fourier transforms
    f1 = np.fft.fftn(data1)
    f2 = np.fft.fftn(data2)
    
    # Cross-spectral density
    cross = f1 * np.conj(f2)
    p1 = np.real(f1 * np.conj(f1))
    p2 = np.real(f2 * np.conj(f2))
    
    # Calculate radial bins
    nz, ny, nx = data1.shape
    kz = np.fft.fftfreq(nz, d=voxel_size[2])
    ky = np.fft.fftfreq(ny, d=voxel_size[1])
    kx = np.fft.fftfreq(nx, d=voxel_size[0])
    
    Kz, Ky, Kx = np.meshgrid(kz, ky, kx, indexing='ij')
    k = np.sqrt(Kz**2 + Ky**2 + Kx**2)
    
    # Flatten everything
    k = k.ravel()
    cross = cross.ravel()
    p1 = p1.ravel()
    p2 = p2.ravel()
    
    # Sort by frequency
    idx = np.argsort(k)
    k_sorted = k[idx]
    cross_sorted = cross[idx]
    p1_sorted = p1[idx]
    p2_sorted = p2[idx]
    
    # Binning
    n_bins = min(nx, ny, nz) // 2
    bins = np.linspace(0, k_sorted.max(), n_bins + 1)
    
    fsc = []
    freqs = []
    
    for i in range(n_bins):
        mask = (k_sorted >= bins[i]) & (k_sorted < bins[i+1])
        if np.any(mask):
            c_bin = cross_sorted[mask]
            p1_bin = p1_sorted[mask]
            p2_bin = p2_sorted[mask]
            
            # Sum of cross power and individual powers
            sum_cross = np.sum(c_bin)
            sum_p1 = np.sum(p1_bin)
            sum_p2 = np.sum(p2_bin)
            
            # FSC is real part of cross correlation / sqrt(power1 * power2)
            # Standard definition uses the real part of the sum
            num = np.real(sum_cross)
            den = np.sqrt(sum_p1 * sum_p2)
            
            if den > 0:
                fsc.append(num / den)
                freqs.append((bins[i] + bins[i+1]) / 2.0)
                
    return np.array(freqs), np.array(fsc)

def compute_ccc(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Compute the Cross-Correlation Coefficient (CCC) between two 3D maps.

    Args:
        data1 (numpy.ndarray): The first 3D density map.
        data2 (numpy.ndarray): The second 3D density map.

    Returns:
        float: The Pearson cross-correlation coefficient between the two maps.
    """
    assert data1.shape == data2.shape
    
    # Flatten and remove mean
    d1 = data1.ravel()
    d2 = data2.ravel()
    
    d1 = d1 - np.mean(d1)
    d2 = d2 - np.mean(d2)
    
    num = np.sum(d1 * d2)
    den = np.sqrt(np.sum(d1**2) * np.sum(d2**2))
    
    if den == 0:
        return 0.0
    return num / den

def save_mrc(data: np.ndarray, output_path: str, origin: tuple[float, float, float] = (0.0, 0.0, 0.0), spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
    """
    Save numpy array to an MRC file.

    Args:
        data (numpy.ndarray): The 3D density map to save.
        output_path (str): Path to the output MRC file.
        origin (tuple, optional): The origin coordinates (x, y, z). Defaults to (0, 0, 0).
        spacing (tuple, optional): The voxel size in Angstroms (x, y, z). Defaults to (1, 1, 1).
    """
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = spacing
        # mrcfile uses x, y, z for origin
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]
        mrc.update_header_from_data()
