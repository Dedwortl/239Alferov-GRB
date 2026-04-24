"""
Analiz effektivnosti metoda zateniya: gruppirovki ot 1 do 12 sputnikov.
Dlya N sputnikov perebirayutsya VSE 2^N konfiguratsiy registratsii.
Konfiguratsii s beloy zonoy = 0 isklyuchayutsya (fizicheski nevozmozhnie).
Po ostavshimsya schitaetsya srednyaya, minimalnaya i maximalnaya belaya zona.
"""

import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, EarthSatellite
import requests
from datetime import timedelta
import itertools


class EarthShadowAnalyzer:

    def __init__(self, norad_id=64881):
        self.R_earth   = 6371.0
        self.norad_id  = norad_id
        self.satellite = None
        self.ts        = None

    def get_tle(self):
        url = (f"https://celestrak.org/NORAD/elements/gp.php"
               f"?CATNR={self.norad_id}&FORMAT=TLE")
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
            print("TLE ne poluchen.")
            return False
        self.ts        = load.timescale()
        self.satellite = EarthSatellite(l1, l2, name, self.ts)
        return True

    def _shadow_center(self, t):
        pos  = self.satellite.at(t).position.km
        dist = np.linalg.norm(pos)
        d    = -pos / dist
        ra   = np.arctan2(d[1], d[0]) % (2 * np.pi)
        dec  = np.arcsin(np.clip(d[2], -1.0, 1.0))
        return ra, dec, dist

    def _shadow_radius(self, dist):
        return float(np.arcsin(self.R_earth / max(dist, self.R_earth + 1)))

    def _make_grid(self, nx=600, ny=300):
        """
        Returns RA, DEC, W all shape (ny, nx).
        W[i,j] = solid angle of pixel in steradians.
        total = sum(W) ~ 4*pi.
        """
        ra_1d  = np.linspace(0, 2 * np.pi, nx, endpoint=False)
        dec_1d = np.linspace(-np.pi / 2, np.pi / 2, ny)
        RA, DEC = np.meshgrid(ra_1d, dec_1d)      # both (ny, nx)
        d_ra  = 2 * np.pi / nx
        d_dec = np.pi / ny
        W     = np.cos(DEC) * d_ra * d_dec        # (ny, nx)
        total = float(W.sum())
        return RA, DEC, W, total

    def _shadow_mask(self, RA, DEC, ra0, dec0, radius):
        cos_d = (np.sin(DEC) * np.sin(dec0)
                 + np.cos(DEC) * np.cos(dec0) * np.cos(RA - ra0))
        return np.arccos(np.clip(cos_d, -1.0, 1.0)) <= radius

    def _white_fraction(self, RA, DEC, W, total, sats, detections):
        """
        True  = caught -> source outside shadow -> remove shadow
        False = missed -> source inside shadow  -> keep only shadow
        Returns sky fraction [0, 1].
        """
        region = np.ones(RA.shape, dtype=bool)    # (ny, nx)
        for s, det in zip(sats, detections):
            mask = self._shadow_mask(RA, DEC, s['ra'], s['dec'], s['radius'])
            if det:
                region &= ~mask
            else:
                region &= mask
        return float(W[region].sum()) / total     # W and region both (ny, nx)

    def _positions(self, base_time, n):
        period_h = 24.0 / 15.2
        sats = []
        for i in range(n):
            dt = timedelta(hours=i * period_h / n)
            t  = self.ts.utc(base_time.utc_datetime() + dt)
            ra, dec, dist = self._shadow_center(t)
            sats.append(dict(
                ra=ra, dec=dec,
                radius=self._shadow_radius(dist),
                phase=round(i * 360 / n),
            ))
        return sats

    def run(self, base_time=None):
        if self.satellite is None:
            if not self.initialize():
                return

        if base_time is None:
            base_time = self.ts.now()

        print("\n" + "=" * 72)
        print(f"Sputnik NORAD {self.norad_id}  |  "
              f"{base_time.utc_strftime('%d.%m.%Y %H:%M UTC')}")
        print("=" * 72)

        RA, DEC, W, total = self._make_grid()

        print(f"{'N':>3}  {'2^N':>6}  {'valid':>9}  "
              f"{'avg %':>10}  {'min %':>8}  {'max %':>8}")
        print("-" * 72)

        counts   = list(range(1, 13))
        avg_list = []
        min_list = []
        max_list = []

        for n in counts:
            sats  = self._positions(base_time, n)
            valid = []

            for det_tuple in itertools.product([True, False], repeat=n):
                frac = self._white_fraction(RA, DEC, W, total,
                                            sats, list(det_tuple))
                if frac > 1e-6:
                    valid.append(frac * 100.0)

            avg = float(np.mean(valid))
            mn  = float(np.min(valid))
            mx  = float(np.max(valid))

            avg_list.append(avg)
            min_list.append(mn)
            max_list.append(mx)

            print(f"{n:>3}  {2**n:>6}  {len(valid):>9}  "
                  f"{avg:>10.2f}  {mn:>8.2f}  {mx:>8.2f}")

        print("=" * 72)
        self._plot(counts, avg_list, min_list, max_list, base_time)

    def _plot(self, counts, avg_list, min_list, max_list, base_time):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11), dpi=150)
        fig.patch.set_facecolor('white')

        # --- верхняя панель: линейный тренд ---
        ax1.fill_between(counts, min_list, max_list,
                         color='steelblue', alpha=0.13,
                         label='диапазон (мин – макс)')
        ax1.plot(counts, avg_list, 'o-', color='steelblue',
                 lw=2.5, ms=8, zorder=5,
                 label='среднее по допустимым конфигурациям')
        ax1.plot(counts, min_list, 's--', color='seagreen',
                 lw=1.8, ms=6, zorder=4,
                 label='минимум (наилучшая конфигурация)')
        ax1.plot(counts, max_list, '^--', color='firebrick',
                 lw=1.8, ms=6, zorder=4,
                 label='максимум (наихудшая конфигурация)')

        for n, avg in zip(counts, avg_list):
            ax1.annotate(f"{avg:.1f}%",
                         xy=(n, avg),
                         xytext=(0, 11), textcoords='offset points',
                         ha='center', fontsize=7.5, color='steelblue',
                         fontweight='bold')

        ax1.set_xticks(counts)
        ax1.set_xticklabels([str(n) for n in counts])
        ax1.set_xlabel('Число спутников в группировке', fontsize=11)
        ax1.set_ylabel('Белая зона (% неба)', fontsize=11)
        ax1.set_title(
            'Средняя белая зона в зависимости от числа спутников\n'
            '(усреднение по всем конфигурациям с ненулевой белой зоной)',
            fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.35)
        ax1.set_facecolor('#f7f7f7')

        # --- нижняя панель: столбчатая диаграмма ---
        w = 0.25
        x = np.array(counts, dtype=float)

        ax2.bar(x - w, avg_list, width=w, color='steelblue', alpha=0.85,
                label='среднее')
        ax2.bar(x,     min_list, width=w, color='seagreen',  alpha=0.85,
                label='минимум')
        ax2.bar(x + w, max_list, width=w, color='firebrick', alpha=0.85,
                label='максимум')

        for n, avg, mn, mx in zip(counts, avg_list, min_list, max_list):
            ax2.text(n - w, avg + 0.3, f"{avg:.1f}", ha='center',
                     fontsize=6, color='steelblue', fontweight='bold')
            ax2.text(n,     mn  + 0.3, f"{mn:.1f}",  ha='center',
                     fontsize=6, color='seagreen',  fontweight='bold')
            ax2.text(n + w, mx  + 0.3, f"{mx:.1f}",  ha='center',
                     fontsize=6, color='firebrick', fontweight='bold')

        ax2.set_xticks(counts)
        ax2.set_xticklabels([str(n) for n in counts])
        ax2.set_xlabel('Число спутников в группировке', fontsize=11)
        ax2.set_ylabel('Белая зона (% неба)', fontsize=11)
        ax2.set_title(
            'Белая зона: среднее, минимум и максимум по всем конфигурациям',
            fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, axis='y', alpha=0.35)
        ax2.set_facecolor('#f7f7f7')

        fig.suptitle(
            f"Эффективность метода затенения  |  NORAD {self.norad_id}  |  "
            f"{base_time.utc_strftime('%d.%m.%Y %H:%M UTC')}",
            fontsize=13, fontweight='bold', y=1.005)

        plt.tight_layout()
        fname = 'efficiency_analysis.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"\nГрафик сохранён: {fname}")


def main():
    a  = EarthShadowAnalyzer(64881)
    ts = load.timescale()
    a.run(ts.now())


if __name__ == "__main__":
    main()
