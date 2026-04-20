import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, EarthSatellite
import requests
from datetime import timedelta
import itertools


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

    def ra_to_mollweide(self, ra):
        x = np.pi - ra
        x = (x + np.pi) % (2 * np.pi) - np.pi
        return x

    # ------------------------------
    # Позиции спутников
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
            t = self.ts.utc(base_time.utc_datetime() + timedelta(hours=shift))
            ra, dec, dist = self.get_shadow_center_radec(t)
            r = self.get_shadow_radius(dist)
            sats.append({
                "ra": ra, "dec": dec, "radius": r,
                "id": i + 1, "phase": i * 90, "time": t
            })
        return sats

    # ------------------------------
    # Вспомогательные методы
    # ------------------------------

    def _make_grid(self):
        ra  = np.linspace(-np.pi, np.pi, 1200)
        dec = np.linspace(-np.pi / 2, np.pi / 2, 800)
        RA, DEC = np.meshgrid(ra, dec)
        ra_equ = (np.pi - RA) % (2 * np.pi)
        return RA, DEC, ra_equ

    def _shadow_mask(self, DEC, ra_equ, ra0, dec0, radius):
        cos_d = (
            np.sin(DEC) * np.sin(dec0)
            + np.cos(DEC) * np.cos(dec0) * np.cos(ra_equ - ra0)
        )
        return np.arccos(np.clip(cos_d, -1, 1)) <= radius

    def _clean_ax(self, ax):
        ax.grid(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # ------------------------------
    # Площадь белой зоны
    # ------------------------------

    def _calc_area(self, sats, detections):
        RA, DEC, ra_equ = self._make_grid()
        combined = np.ones(RA.shape, dtype=bool)
        for sat, det in zip(sats, detections):
            in_shadow = self._shadow_mask(DEC, ra_equ,
                                          sat['ra'], sat['dec'], sat['radius'])
            white = ~in_shadow if det else in_shadow
            combined &= white
        n_dec, n_ra = RA.shape
        dec_vals = np.linspace(-np.pi / 2, np.pi / 2, n_dec)
        weights  = np.cos(dec_vals)[:, np.newaxis] * (2*np.pi/n_ra) * (np.pi/n_dec)
        area_sr  = np.sum(combined * weights)
        return area_sr * (180/np.pi)**2, area_sr / (4*np.pi)

    # ------------------------------
    # Карта одного спутника
    # ------------------------------

    def _draw_single(self, ax, sat, detected):
        """
        detected=True  → тень чёрная (там сигнал не мог прийти)
        detected=False → вне тени чёрная (там сигнал прошёл бы)
        """
        RA, DEC, ra_equ = self._make_grid()
        in_shadow = self._shadow_mask(DEC, ra_equ,
                                      sat['ra'], sat['dec'], sat['radius'])

        if detected:
            ax.set_facecolor('white')
            ax.contourf(RA, DEC, in_shadow.astype(float),
                        levels=[0.5, 1.5], colors=['black'], alpha=1.0)
        else:
            ax.set_facecolor('black')
            ax.contourf(RA, DEC, in_shadow.astype(float),
                        levels=[0.5, 1.5], colors=['white'], alpha=1.0)

        self._clean_ax(ax)

        status = "ПОЙМАЛ ✓" if detected else "НЕ ПОЙМАЛ ✗"
        color  = '#007722' if detected else '#cc1111'
        ax.set_title(
            f"Спутник {sat['id']} (фаза {sat['phase']}°)\n"
            f"RA={np.degrees(sat['ra']):.1f}°  "
            f"Dec={np.degrees(sat['dec']):.1f}°\n"
            f"{status}",
            fontsize=7, color=color, fontweight='bold', pad=3
        )

    # ------------------------------
    # Итоговая карта
    # ------------------------------

    def _draw_combined(self, ax, sats, detections, area_deg2, fraction):
        """
        Белое = пересечение всех белых зон (возможное положение источника).
        Накладываем чёрные слои для каждого спутника.
        """
        RA, DEC, ra_equ = self._make_grid()

        ax.set_facecolor('white')

        for sat, det in zip(sats, detections):
            in_shadow = self._shadow_mask(DEC, ra_equ,
                                          sat['ra'], sat['dec'], sat['radius'])
            if det:
                # тень — чёрная
                ax.contourf(RA, DEC, in_shadow.astype(float),
                            levels=[0.5, 1.5], colors=['black'], alpha=1.0)
            else:
                # вне тени — чёрная
                outside = (~in_shadow).astype(float)
                ax.contourf(RA, DEC, outside,
                            levels=[0.5, 1.5], colors=['black'], alpha=1.0)

        self._clean_ax(ax)
        ax.set_title(
            f"ИТОГ: возможное положение источника\n"
            f"Белая площадь: {area_deg2:.0f} кв.°  "
            f"({fraction*100:.2f}% неба)",
            fontsize=7, color='black', fontweight='bold', pad=3
        )

    # ------------------------------
    # 16 конфигураций
    # ------------------------------

    def plot_all_configurations(self, sats, current_time):
        all_configs = list(itertools.product([True, False], repeat=4))

        for cfg_idx, detections in enumerate(all_configs):
            area_deg2, fraction = self._calc_area(sats, detections)

            labels       = [f"Сп{i+1}:{'✓' if d else '✗'}"
                            for i, d in enumerate(detections)]
            config_label = "  ".join(labels)
            binary_str   = "".join(['1' if d else '0' for d in detections])

            fig, axes = plt.subplots(
                1, 5, figsize=(22, 4), dpi=120,
                subplot_kw={'projection': 'mollweide'}
            )
            fig.patch.set_facecolor('white')

            for i, (ax, det) in enumerate(zip(axes[:4], detections)):
                self._draw_single(ax, sats[i], det)

            self._draw_combined(axes[4], sats, detections, area_deg2, fraction)

            time_str = current_time.utc_strftime('%d.%m.%Y %H:%M UTC')
            fig.suptitle(
                f"Конфигурация #{cfg_idx+1:02d}  [{binary_str}]   {config_label}\n"
                f"Спутник {self.norad_id}  |  {time_str}  |  "
                f"1=поймал сигнал, 0=не поймал",
                fontsize=10, fontweight='bold', color='black', y=1.02
            )

            plt.tight_layout()

            filename = f"config_{cfg_idx+1:02d}_{binary_str}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            print(f"  Сохранено: {filename}  |  {config_label}"
                  f"  |  Белая область: {area_deg2:.0f} кв.°")

        print("\nВсего сохранено 16 файлов.")

    # ------------------------------
    # Главный метод
    # ------------------------------

    def run_all_16(self, current_time=None):
        if self.satellite is None:
            if not self.initialize():
                return

        if current_time is None:
            current_time = self.ts.now()

        print("\n" + "=" * 60)
        print(f"Спутник NORAD {self.norad_id}")
        print(f"Время: {current_time.utc_strftime('%d.%m.%Y %H:%M UTC')}")
        print("=" * 60)
        print("Расчёт 16 конфигураций регистрации гамма-всплеска...\n")

        sats = self.calculate_satellite_positions(current_time)
        if not sats:
            print("Не удалось получить положения спутников.")
            return

        print("Положения спутников:")
        for s in sats:
            print(f"  Фаза {s['phase']}°: RA={np.degrees(s['ra']):.2f}°, "
                  f"Dec={np.degrees(s['dec']):.2f}°, "
                  f"Радиус затенения={np.degrees(s['radius']):.2f}°")
        print()

        self.plot_all_configurations(sats, current_time)
        plt.show()


# ------------------------------
# main
# ------------------------------

def main():
    v = EarthShadowVisualizer(64881)
    ts = load.timescale()
    t  = ts.now()
    v.run_all_16(t)


if __name__ == "__main__":
    main()