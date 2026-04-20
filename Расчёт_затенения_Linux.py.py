import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, EarthSatellite
import requests
from datetime import timedelta
import mhealpy as hmap
from mhealpy import HealpixMap
import copy


class EarthShadowVisualizer:

    def __init__(self, norad_id=64881):
        self.R_earth = 6371.0
        self.norad_id = norad_id
        self.satellite = None
        self.ts = None

    def get_tle(self):
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={self.norad_id}&FORMAT=TLE"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                lines = r.text.strip().split("\n")
                if len(lines) >= 3:
                    return lines[0], lines[1], lines[2]
        except:
            pass
        return None, None, None

    def initialize(self):
        name, l1, l2 = self.get_tle()
        if name is None:
            print("TLE не получен")
            return False
        self.ts = load.timescale()
        self.satellite = EarthSatellite(l1, l2, name, self.ts)
        return True

    def get_shadow_center_radec(self, t):
        geocentric = self.satellite.at(t)
        pos = geocentric.position.km
        dist = np.linalg.norm(pos)
        direction = -pos / dist
        x, y, z = direction
        ra = np.arctan2(y, x)
        if ra < 0:
            ra += 2 * np.pi
        dec = np.arcsin(z)
        return ra, dec, dist

    def get_shadow_radius(self, distance):
        if distance <= self.R_earth:
            return np.pi / 2
        return np.arcsin(self.R_earth / distance)

    def calculate_satellite_positions(self, base_time):
        if self.satellite is None:
            if not self.initialize():
                return []
        mean_motion = 15.2
        period_hours = 24 / mean_motion
        sats = []
        for i in range(4):
            shift = i * period_hours / 4
            t = self.ts.utc(base_time.utc_datetime() + timedelta(hours=shift))
            ra, dec, dist = self.get_shadow_center_radec(t)
            r = self.get_shadow_radius(dist)
            sats.append({
                "ra": ra,
                "dec": dec,
                "radius": r,
                "id": i + 1,
                "phase": i * 90,
                "time": t
            })
        return sats

    def create_occultation_maps(self, sats, nside=64):
        lst_maps = []
        for sat in sats:
            m = HealpixMap(nside=nside, scheme='ring', density=True)
            m[:] = 1
            
            # Преобразование RA, Dec в theta, phi для HEALPix
            # theta = 90° - Dec (полярный угол от северного полюса)
            theta = np.deg2rad(90 - np.degrees(sat["dec"]))
            # phi = RA (но нужно учесть поворот на 180°, как в occ.py)
            phi = sat["ra"]
            
            # КЛЮЧЕВОЕ: поворот на 180°, как в occ.py
            if phi < np.pi:
                phi += np.pi
            else:
                phi -= np.pi
            
            radius = sat["radius"]
            center_vec = hmap.ang2vec(theta, phi)
            disc_pix = m.query_disc(center_vec, radius)
            m[disc_pix] = 0
            lst_maps.append(m)
        
        m_tot = HealpixMap(nside=nside, scheme='ring', density=True)
        m_tot[:] = 1
        return lst_maps, m_tot

    def compute_unocculted_fraction(self, sats, nside=64):
        list_maps, m_tot = self.create_occultation_maps(sats, nside)
        mRes = copy.deepcopy(list_maps[0])
        for occ in list_maps[1:]:
            mRes *= occ
        sum_unocc = np.sum(mRes)
        sum_tot = np.sum(m_tot)
        fraction = sum_unocc / sum_tot
        occulted_fraction = 1 - fraction
        return fraction, occulted_fraction, mRes, list_maps, m_tot

    def plot_healpix_maps(self, sats, mRes, list_maps, m_tot, fraction, occulted_fraction, current_time):
        n_maps = len(list_maps)
        fig, axes = plt.subplots(n_maps + 1, 1, 
                                  figsize=(12, 4 * (n_maps + 1)),
                                  dpi=150,
                                  subplot_kw={'projection': 'mollview'})
        if n_maps + 1 == 1:
            axes = [axes]
        for i, ax in enumerate(axes[:-1]):
            ax.set_title(f"Спутник {sats[i]['id']} (фаза {sats[i]['phase']}°)\n"
                        f"RA={np.degrees(sats[i]['ra']):.1f}°, Dec={np.degrees(sats[i]['dec']):.1f}°, "
                        f"Радиус={np.degrees(sats[i]['radius']):.1f}°", size=9)
        axes[-1].set_title(f"Результат перемножения (незатенённые области)\n"
                          f"Незатенённая доля: {fraction*100:.2f}% | "
                          f"Затенённая доля: {occulted_fraction*100:.2f}%", size=10)
        for i, m in enumerate(list_maps):
            m.plot(axes[i], cbar=None, cmap='gray')
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
        mRes.plot(axes[-1], cbar=None, cmap='gray')
        axes[-1].set_xlabel("")
        axes[-1].set_ylabel("")
        time_str = current_time.utc_strftime('%d.%m.%Y %H:%M UTC')
        fig.suptitle(f'Спутник {self.norad_id} | {time_str}\n'
                     f'Доля незатенённого неба с учётом пересечений: {fraction*100:.2f}%', 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig

    def analyze_occultation(self, current_time=None):
        if self.satellite is None:
            if not self.initialize():
                return None, None
        if current_time is None:
            current_time = self.ts.now()
        sats = self.calculate_satellite_positions(current_time)
        if not sats:
            return None, None
        print("\n" + "="*60)
        print(f"Анализ затенения для спутника {self.norad_id}")
        print(f"Время: {current_time.utc_strftime('%d.%m.%Y %H:%M UTC')}")
        print("="*60)
        print("\nПоложения спутника для 4 фаз орбиты:")
        for sat in sats:
            print(f"  Фаза {sat['phase']}°: RA={np.degrees(sat['ra']):.2f}°, "
                  f"Dec={np.degrees(sat['dec']):.2f}°, "
                  f"Радиус затенения={np.degrees(sat['radius']):.2f}°")
        fraction, occulted_fraction, _, _, _ = self.compute_unocculted_fraction(sats)
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ РАСЧЁТА:")
        print(f"  Доля незатенённого неба: {fraction*100:.4f}%")
        print(f"  Доля затенённого неба: {occulted_fraction*100:.4f}%")
        print("="*60)
        return fraction, occulted_fraction

    def plot_shadow_regions_healpix(self, current_time=None):
        if self.satellite is None:
            if not self.initialize():
                return None
        if current_time is None:
            current_time = self.ts.now()
        sats = self.calculate_satellite_positions(current_time)
        if not sats:
            return None
        fraction, occulted_fraction, mRes, list_maps, m_tot = self.compute_unocculted_fraction(sats)
        fig = self.plot_healpix_maps(sats, mRes, list_maps, m_tot, fraction, occulted_fraction, current_time)
        return fig, fraction, occulted_fraction


def main():
    v = EarthShadowVisualizer(64881)
    ts = load.timescale()
    t = ts.now()
    
    fraction, occulted_fraction = v.analyze_occultation(t)
    
    fig, fraction, occulted_fraction = v.plot_shadow_regions_healpix(t)
    
    if fig:
        filename = f"earth_shadow_healpix_{t.utc_strftime('%Y%m%d_%H%M')}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\nГрафик сохранён: {filename}")
        plt.show()


if __name__ == "__main__":
    main()
