import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, EarthSatellite
import requests
from datetime import timedelta


class EarthShadowVisualizer:

    def __init__(self, norad_id=64881):

        self.R_earth = 6371.0
        self.norad_id = norad_id
        self.satellite = None
        self.ts = None

    def get_tle(self):
        """Получение TLE"""
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
            print("TLE не получен, возможно ошибка сети.")
            return False

        self.ts = load.timescale()

        self.satellite = EarthSatellite(l1, l2, name, self.ts)

        return True

    def get_shadow_center_radec(self, t):
        #Позиция спутника относительно центра Земли (вектор)
        geocentric = self.satellite.at(t)

        pos = geocentric.position.km

        dist = np.linalg.norm(pos)
        #Направление на центр Земли (от спутника)
        direction = - pos / dist

        x, y, z = direction
        #Перевод в RA, Dec
        ra = np.arctan2(y, x)

        if ra < 0:
            ra += 2 * np.pi

        dec = np.arcsin(z)

        return ra, dec, dist

    def get_shadow_radius(self, distance):

        if distance <= self.R_earth:
            return np.pi / 2

        return np.arcsin(self.R_earth / distance)

    def fill_shadow_region(self, ax, ra0, dec0, radius, color):
        # Повышаем разрешение
        ra = np.linspace(0, 24, 1200)
        dec = np.linspace(-90, 90, 800)

        RA, DEC = np.meshgrid(ra, dec)

        ra_rad = RA * np.pi / 12
        dec_rad = np.radians(DEC)

        cos_d = (
                np.sin(dec_rad) * np.sin(dec0) +
                np.cos(dec_rad) * np.cos(dec0) * np.cos(ra_rad - ra0)
        )
        cos_d = np.clip(cos_d, -1, 1)
        ang = np.arccos(cos_d)

        mask = ang <= radius

        # контур и заливка
        ax.contourf(
            RA, DEC, mask,
            levels=[0.5, 1],
            colors=[color],
            alpha=0.25
        )

    def generate_shadow_contour(self, ra0, dec0, radius, n=1000):
        beta = np.linspace(0, 2 * np.pi, n)
        dec = np.arcsin(
            np.sin(dec0) * np.cos(radius) +
            np.cos(dec0) * np.sin(radius) * np.cos(beta)
        )
        cos_dra = (np.cos(radius) - np.sin(dec0) * np.sin(dec)) / (np.cos(dec0) * np.cos(dec))
        cos_dra = np.clip(cos_dra, -1, 1)
        dra = np.arccos(cos_dra)

        dra = dra * np.sign(np.sin(beta))

        ra = ra0 + dra
        ra = ra % (2 * np.pi)

        ra_hours = ra * 12 / np.pi
        dec_deg = np.degrees(dec)
        segments = []
        current_segment_ra = []
        current_segment_dec = []

        for i in range(len(ra_hours)):
            if i == 0:
                current_segment_ra.append(ra_hours[i])
                current_segment_dec.append(dec_deg[i])
            else:
                diff = abs(ra_hours[i] - ra_hours[i - 1])
                if diff > 10:
                    if current_segment_ra:
                        segments.append((np.array(current_segment_ra), np.array(current_segment_dec)))
                        current_segment_ra = []
                        current_segment_dec = []

                current_segment_ra.append(ra_hours[i])
                current_segment_dec.append(dec_deg[i])

        # Добавляем последний сегмент
        if current_segment_ra:
            segments.append((np.array(current_segment_ra), np.array(current_segment_dec)))

        if not segments:
            return np.array([]), np.array([])

        all_ra = []
        all_dec = []

        for j, (seg_ra, seg_dec) in enumerate(segments):
            if j > 0:
                # Добавляем NaN для создания разрыва
                all_ra.append(np.nan)
                all_dec.append(np.nan)
            all_ra.extend(seg_ra)
            all_dec.extend(seg_dec)

        if len(segments) == 1 and len(segments[0][0]) > 2:
            # Добавляем первую точку в конец для замыкания
            all_ra.append(segments[0][0][0])
            all_dec.append(segments[0][1][0])
        elif len(segments) > 1:
            if len(segments[0][0]) > 0 and len(segments[-1][0]) > 0:
                all_ra.append(np.nan)
                all_dec.append(np.nan)
                all_ra.append(segments[0][0][0])
                all_dec.append(segments[0][1][0])

        return np.array(all_ra), np.array(all_dec)

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
                "phase": i * 90

            })

        return sats

    def plot_shadow_regions(self, current_time=None):

        if self.satellite is None:

            if not self.initialize():
                return

        if current_time is None:
            current_time = self.ts.now()

        sats = self.calculate_satellite_positions(current_time)

        fig, ax = plt.subplots(figsize=(12, 10))

        ax.set_xlim(24, 0)
        ax.set_ylim(-90, 90)

        ax.set_xlabel("Прямое восхождение (часы)", fontsize=12)
        ax.set_ylabel("Склонение (градусы)", fontsize=12)

        ax.set_title("Области затенения неба Землей\nв экваториальных координатах",
                     fontsize=14, fontweight='bold')

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)  # Небесный экватор

        colors = ["red", "blue", "green", "orange"]

        for i, s in enumerate(sats):

            self.fill_shadow_region(
                ax,
                s["ra"],
                s["dec"],
                s["radius"],
                colors[i]
            )

            ra_h, dec_d = self.generate_shadow_contour(
                s["ra"],
                s["dec"],
                s["radius"]
            )

            if len(ra_h) > 0:
                ax.plot(
                    ra_h,
                    dec_d,
                    color=colors[i],
                    linewidth=2,
                    label=f'Спутник {s["id"]} (фаза {s["phase"]}°)'
                )

            ax.plot(
                s["ra"] * 12 / np.pi,
                np.degrees(s["dec"]),
                "o",
                color=colors[i],
                markersize=8,
                markeredgecolor='black',
                markeredgewidth=1
            )

        ax.legend(loc='upper right', fontsize=10)

        time_str = current_time.utc_strftime('%d.%m.%Y %H:%M UTC')
        fig.suptitle(f'239Alferov (NORAD 64881) | {time_str}', fontsize=12)

        plt.tight_layout()

        return fig


def main():
    v = EarthShadowVisualizer(64881)

    ts = load.timescale()

    t = ts.now()

    fig = v.plot_shadow_regions(t)

    if fig:
        plt.show()

        fig.savefig(
            "earth_shadow_radec.png",
            dpi=300,
            bbox_inches="tight"
        )
        print("График сохранен: earth_shadow_radec.png")


if __name__ == "__main__":
    main()