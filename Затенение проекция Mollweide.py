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

    # ------------------------------
    # TLE
    # ------------------------------

    def get_tle(self):
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={self.norad_id}&FORMAT=TLE"

        try:
            r = requests.get(url, timeout=10)

            if r.status_code == 200:
                lines = r.text.strip().split("\n")

                if len(lines) >= 3:
                    return lines[0], lines[1], lines[2]

        except Exception:
            pass

        return None, None, None

    # ------------------------------

    def initialize(self):

        name, l1, l2 = self.get_tle()

        if name is None:
            print("TLE не получен")
            return False

        self.ts = load.timescale()
        self.satellite = EarthSatellite(l1, l2, name, self.ts)

        return True

    # ------------------------------
    # Геометрия тени
    # ------------------------------

    def get_shadow_center_radec(self, t):

        geocentric = self.satellite.at(t)

        pos = geocentric.position.km
        dist = np.linalg.norm(pos)

        direction = - pos / dist

        x, y, z = direction

        ra = np.arctan2(y, x)

        if ra < 0:
            ra += 2 * np.pi

        dec = np.arcsin(z)

        return ra, dec, dist

    # ------------------------------

    def get_shadow_radius(self, distance):

        if distance <= self.R_earth:
            return np.pi / 2

        return np.arcsin(self.R_earth / distance)

    # ------------------------------
    # Перевод координат в Mollweide
    # ------------------------------

    def ra_to_mollweide(self, ra):

        x = np.pi - ra
        x = (x + np.pi) % (2 * np.pi) - np.pi

        return x

    # ------------------------------
    # Заполнение области
    # ------------------------------

    def fill_shadow_region(self, ax, ra0, dec0, radius, color):

        ra = np.linspace(-np.pi, np.pi, 1200)
        dec = np.linspace(-np.pi / 2, np.pi / 2, 800)

        RA, DEC = np.meshgrid(ra, dec)

        ra_equ = np.pi - RA
        ra_equ = ra_equ % (2 * np.pi)

        cos_d = (
                np.sin(DEC) * np.sin(dec0)
                + np.cos(DEC) * np.cos(dec0) * np.cos(ra_equ - ra0)
        )

        cos_d = np.clip(cos_d, -1, 1)

        ang = np.arccos(cos_d)

        mask = ang <= radius

        ax.contourf(
            RA,
            DEC,
            mask,
            levels=[0.5, 1],
            colors=[color],
            alpha=0.25
        )

    # ------------------------------
    # Контур тени
    # ------------------------------

    def generate_shadow_contour(self, ra0, dec0, radius, n=2000):

        beta = np.linspace(0, 2 * np.pi, n)

        dec = np.arcsin(
            np.sin(dec0) * np.cos(radius)
            + np.cos(dec0) * np.sin(radius) * np.cos(beta)
        )

        # защита от полюсов
        den = np.cos(dec0) * np.cos(dec)
        den = np.where(np.abs(den) < 1e-10, np.nan, den)

        cos_dra = (np.cos(radius) - np.sin(dec0) * np.sin(dec)) / den

        cos_dra = np.clip(cos_dra, -1, 1)

        dra = np.arccos(cos_dra)

        dra *= np.sign(np.sin(beta))

        ra = (ra0 + dra) % (2 * np.pi)

        ra = self.ra_to_mollweide(ra)

        ra = np.clip(ra, -np.pi, np.pi)

        # -----------------------
        # разрыв через край карты
        # -----------------------

        diff = np.abs(np.diff(ra))

        breaks = diff > np.pi / 2

        ra_new = [ra[0]]
        dec_new = [dec[0]]

        for i in range(1, len(ra)):

            if breaks[i - 1]:
                ra_new.append(np.nan)
                dec_new.append(np.nan)

            ra_new.append(ra[i])
            dec_new.append(dec[i])

        return np.array(ra_new), np.array(dec_new)

    # ------------------------------
    # позиции спутников
    # ------------------------------

    def calculate_satellite_positions(self, base_time):

        if self.satellite is None:
            if not self.initialize():
                return []

        mean_motion = 15.2
        period_hours = 24 / mean_motion

        sats = []

        for i in range(4):

            shift = i * period_hours / 4

            t = self.ts.utc(
                base_time.utc_datetime() + timedelta(hours=shift)
            )

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

    # ------------------------------
    # График
    # ------------------------------

    def plot_shadow_regions(self, current_time=None):

        if self.satellite is None:
            if not self.initialize():
                return

        if current_time is None:
            current_time = self.ts.now()

        sats = self.calculate_satellite_positions(current_time)

        fig = plt.figure(figsize=(12, 10))

        ax = fig.add_subplot(111, projection="mollweide")

        ax.grid(True, linestyle="--", alpha=0.3)

        # подписи осей
        ax.set_xlabel("Прямое восхождение (часы)", fontsize=12)
        ax.set_ylabel("Склонение (градусы)", fontsize=12)

        # RA ticks
        xticks = np.radians(np.linspace(-150, 150, 11))
        xticklabels = [
            "22h", "20h", "18h", "16h", "14h",
            "12h",
            "10h", "8h", "6h", "4h", "2h"
        ]

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # Dec ticks
        yticks = np.radians(np.arange(-75, 90, 15))
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{d}°" for d in np.arange(-75, 90, 15)])

        ax.set_title(
            "Области затенения неба Землей\nв экваториальных координатах",
            fontsize=14,
            fontweight='bold'
        )

        colors = ["red", "blue", "green", "orange"]

        for i, s in enumerate(sats):

            self.fill_shadow_region(
                ax,
                s["ra"],
                s["dec"],
                s["radius"],
                colors[i]
            )

            ra_c, dec_c = self.generate_shadow_contour(
                s["ra"],
                s["dec"],
                s["radius"]
            )

            ax.plot(
                ra_c,
                dec_c,
                color=colors[i],
                linewidth=2,
                label=f'Спутник {s["id"]} (фаза {s["phase"]}°)'
            )

            ax.plot(
                self.ra_to_mollweide(s["ra"]),
                s["dec"],
                "o",
                color=colors[i],
                markersize=8,
                markeredgecolor='black',
                markeredgewidth=1
            )

        ax.legend(loc='upper right', fontsize=10)

        time_str = current_time.utc_strftime('%d.%m.%Y %H:%M UTC')

        fig.suptitle(
            f'239Alferov (NORAD 64881) | {time_str}',
            fontsize=12
        )

        plt.tight_layout()

        return fig


# ------------------------------
# main
# ------------------------------

def main():

    v = EarthShadowVisualizer(64881)

    ts = load.timescale()

    t = ts.now()

    fig = v.plot_shadow_regions(t)

    if fig:

        plt.show()

        fig.savefig(
            "earth_shadow_mollweide.png",
            dpi=300,
            bbox_inches="tight"
        )

        print("График сохранен: earth_shadow_mollweide.png")


if __name__ == "__main__":
    main()