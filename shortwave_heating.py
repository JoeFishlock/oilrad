import numpy as np
import matplotlib.pyplot as plt
from oilrad import calculate_SW_heating_in_ice

if __name__ == "__main__":
    ICE_TYPE = "FYI"
    DSI = 280
    THICKNESS = 0.8

    plt.figure()
    plt.title(f"Incident DSI: {DSI}W/m2, ice type: {ICE_TYPE}, thickness: {THICKNESS}m")
    z = np.linspace(-THICKNESS, 0, 100)
    for oil in [0, 10, 50, 100, 500, 1000]:
        heating = calculate_SW_heating_in_ice(
            DSI,
            z,
            "1L",
            350,
            700,
            oil_mass_ratio=oil,
            ice_thickness=THICKNESS,
            ice_type=ICE_TYPE,
        )
        plt.plot(heating, z, label=f"{oil}ng/g")
    plt.ylabel("Depth in ice (m)")
    plt.xlabel("SW heating (W/m3)")
    plt.legend()
    plt.savefig("figures/shortwave_heating.pdf")
    plt.close()
