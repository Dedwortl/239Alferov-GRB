import math
import numpy as np
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import requests
from skyfield.api import load, EarthSatellite
import matplotlib.dates as mdates

class CubeSatShadowSimple:
    def __init__(self, norad_id=64881):
        self.R_earth = 6371.0
        self.norad_id = norad_id
        self.satellite = None
        self.ts = None
        
    def get_tle(self):
        """Получение TLE"""
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={self.norad_id}&FORMAT=TLE"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                if len(lines) >= 3:
                    return lines[0].strip(), lines[1].strip(), lines[2].strip()
        except:
            pass
        
        # Резервные TLE
        return ("239ALFEROV",
               "1 64881U 22152H   25020.00000000  .00000000  00000-0  00000-0 0  9999",
               "2 64881  97.5000  45.0000 0010000 270.0000  90.0000 15.20000000    15")
    
    def initialize(self):
        """Инициализация спутника"""
        tle = self.get_tle()
        self.ts = load.timescale()
        self.satellite = EarthSatellite(tle[1], tle[2], tle[0], self.ts)
        print(f"Спутник инициализирован: {tle[0]}")
        return True
    
    def calculate_shadow_percentage(self, distance_to_center_km):
        """
        Расчет процента затемнения (какая часть неба занята Землей)
        Формула: % = [1 - cos(arcsin(R/(R+h)))] / 2 * 100%
        
        Args:
            distance_to_center_km: расстояние от центра Земли в км
        """
        R = self.R_earth
        r = distance_to_center_km
        
        # Высота над сферической Землей
        h = r - R
        
        # Угол от надира до горизонта
        α = math.asin(R / r)
        
        # Процент неба, занятый Землей
        percentage = (1 - math.cos(α)) / 2 * 100
        
        return percentage, h
    
    def get_shadow_vs_time(self, points=1000):
        """
        Расчет затемнения за последние 12 часов
        
        Args:
            points: количество точек
        
        Возвращает:
        times - объекты datetime
        shadows - процент затемнения
        distances - расстояния до центра Земли
        geometric_altitudes - геометрические высоты (расстояние - R_earth)
        positions - полные позиции
        """
        if self.satellite is None:
            self.initialize()
        
        # Создаем временной диапазон: последние 12 часов
        hours_back = 12
        start_time = self.ts.now() - hours_back / 24.0
        end_time = self.ts.now()
        times = self.ts.linspace(start_time, end_time, points)
        
        datetimes = []
        shadows = []
        distances = []
        geometric_altitudes = []
        latitudes = []
        longitudes = []
        positions = []
        
        for t in times:
            # Позиция спутника
            geocentric = self.satellite.at(t)
            
            # Получаем прямое расстояние до центра Земли
            ra, dec, distance_km = geocentric.radec()
            
            # Также получаем субспутниковую точку для координат
            subpoint = geocentric.subpoint()
            
            dt = t.utc_datetime()
            lat = subpoint.latitude.degrees
            lon = subpoint.longitude.degrees
            
            # Расчет затемнения и геометрической высоты
            shadow, geo_alt = self.calculate_shadow_percentage(distance_km.km)
            
            # Сохранение данных
            datetimes.append(dt)
            shadows.append(shadow)
            distances.append(distance_km.km)
            geometric_altitudes.append(geo_alt)
            latitudes.append(lat)
            longitudes.append(lon)
            
            positions.append({
                'time': dt,
                'lat': lat,
                'lon': lon,
                'distance_to_center': distance_km.km,
                'geometric_altitude': geo_alt,
                'shadow': shadow
            })
        
        return datetimes, shadows, distances, geometric_altitudes, latitudes, longitudes, positions
    
    def get_shadow_data(self, points=1000):
        """
        Возвращает данные в структурированном виде за последние 12 часов
        
        Returns:
        dict: словарь с данными о затемнении
        """
        datetimes, shadows, distances, geo_alts, lats, lons, positions = self.get_shadow_vs_time(points)
        
        # Вычисляем статистику
        shadow_array = np.array(shadows)
        dist_array = np.array(distances)
        alt_array = np.array(geo_alts)
        
        data = {
            'times': datetimes,
            'shadows': shadows,
            'distances_to_center': distances,
            'geometric_altitudes': geo_alts,
            'latitudes': lats,
            'longitudes': lons,
            'positions': positions,
            'statistics': {
                'shadow': {
                    'mean': float(np.mean(shadow_array)),
                    'std': float(np.std(shadow_array)),
                    'min': float(np.min(shadow_array)),
                    'max': float(np.max(shadow_array)),
                    'median': float(np.median(shadow_array))
                },
                'distance_to_center': {
                    'mean': float(np.mean(dist_array)),
                    'std': float(np.std(dist_array)),
                    'min': float(np.min(dist_array)),
                    'max': float(np.max(dist_array)),
                    'range': float(np.max(dist_array) - np.min(dist_array))
                },
                'geometric_altitude': {
                    'mean': float(np.mean(alt_array)),
                    'std': float(np.std(alt_array)),
                    'min': float(np.min(alt_array)),
                    'max': float(np.max(alt_array)),
                    'range': float(np.max(alt_array) - np.min(alt_array))
                }
            },
            'metadata': {
                'norad_id': self.norad_id,
                'start_time': datetimes[0],
                'end_time': datetimes[-1],
                'duration_hours': 12,
                'points': points,
                'current_time': datetime.now(timezone.utc),
                'earth_radius_km': self.R_earth
            }
        }
        
        return data
    
    def plot_shadow_evolution(self):
        """График затемнения и расстояния до центра Земли за последние 12 часов"""
        datetimes, shadows, distances, geo_alts, lats, lons, positions = self.get_shadow_vs_time(1000)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 16))
        
        # 1. График затемнения (главный)
        ax1 = axes[0]
        ax1.plot(datetimes, shadows, 'b-', linewidth=2, label='% неба закрыто Землей')
        
        # Заливка области отклонений
        shadow_mean = np.mean(shadows)
        shadow_std = np.std(shadows)
        
        ax1.fill_between(datetimes, 
                        shadows, 
                        shadow_mean, 
                        where=shadows > shadow_mean, 
                        alpha=0.3, color='red', label='Выше среднего')
        ax1.fill_between(datetimes, 
                        shadows, 
                        shadow_mean, 
                        where=shadows <= shadow_mean, 
                        alpha=0.3, color='green', label='Ниже среднего')
        
        # Средняя линия и зона ±1σ
        ax1.axhline(shadow_mean, color='black', linestyle='--', linewidth=1.5, 
                   label=f'Среднее: {shadow_mean:.2f}%')
        ax1.axhline(shadow_mean + shadow_std, color='gray', linestyle=':', linewidth=1)
        ax1.axhline(shadow_mean - shadow_std, color='gray', linestyle=':', linewidth=1)
        ax1.fill_between(datetimes, shadow_mean - shadow_std, shadow_mean + shadow_std, 
                        alpha=0.1, color='gray', label='±1σ')
        
        # Настройки графика
        ax1.set_ylabel('Процент затемнения (%)', fontsize=12)
        ax1.set_title('Затемнение неба кубсатом за последние 12 часов', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(max(0, shadow_mean - 3*shadow_std), shadow_mean + 3*shadow_std)
        ax1.legend(loc='upper right', fontsize=9)
        
        # Форматирование оси времени в UTC
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d.%m', tz=timezone.utc))
        
        # Подписи каждые 3 часа
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3, tz=timezone.utc))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Добавляем вертикальную линию для текущего времени
        current_time = datetime.now(timezone.utc)
        ax1.axvline(current_time, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Текущее время\n{current_time.strftime("%H:%M UTC")}')
        
        # 2. График расстояния до центра Земли
        ax2 = axes[1]
        ax2.plot(datetimes, distances, 'r-', linewidth=2, label='Расстояние до центра Земли')
        ax2.fill_between(datetimes, distances, alpha=0.3, color='red')
        
        # Средняя линия расстояния
        dist_mean = np.mean(distances)
        dist_std = np.std(distances)
        ax2.axhline(dist_mean, color='black', linestyle='--', linewidth=1.5, 
                   label=f'Среднее: {dist_mean:.1f} км')
        
        # Показать диапазон
        dist_range = max(distances) - min(distances)
        ax2.text(0.02, 0.95, f'Δr = {dist_range:.1f} км', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Показать теоретические значения апогея и перигея
        apogee_theoretical = self.R_earth + 490  # 6861 км
        perigee_theoretical = self.R_earth + 477  # 6848 км
        ax2.axhline(apogee_theoretical, color='blue', linestyle=':', linewidth=1, 
                   label=f'Теор. апогей: {apogee_theoretical:.0f} км')
        ax2.axhline(perigee_theoretical, color='orange', linestyle=':', linewidth=1,
                   label=f'Теор. перигей: {perigee_theoretical:.0f} км')
        
        ax2.set_ylabel('Расстояние до центра (км)', fontsize=12)
        ax2.set_title('Расстояние до центра Земли', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_ylim(dist_mean - 2*dist_std, dist_mean + 2*dist_std)
        
        
        plt.tight_layout()
        
        return fig
    
    def print_summary(self):
        """Краткая сводка за последние 12 часов"""
        datetimes, shadows, distances, geo_alts, lats, lons, positions = self.get_shadow_vs_time(500)
        
        current_time = datetime.now(timezone.utc)
        
        print("="*70)
        print("РАСЧЕТ ЗАТМЕНИЯ НЕБА ДЛЯ КУБСАТА 239ALFEROV")
        print("(используется расстояние до центра Земли)")
        print("="*70)
        print(f"Текущее время UTC: {current_time.strftime('%d.%m.%Y %H:%M:%S')}")
        print(f"Начало расчета: {datetimes[0].strftime('%d.%m.%Y %H:%M:%S UTC')}")
        print(f"Конец расчета:  {datetimes[-1].strftime('%d.%m.%Y %H:%M:%S UTC')}")
        print(f"Период: последние 12 часов")
        print(f"Радиус Земли (сферический): {self.R_earth} км")
        print()
        
        # Выбор ключевых точек
        print("КЛЮЧЕВЫЕ ТОЧКИ НА ОРБИТЕ:")
        print("-"*70)
        print(f"{'Время UTC':<12} {'Широта':<8} {'Долгота':<9} {'Расст.':<8} {'Высота':<8} {'Земля':<10} {'Космос':<10}")
        print("-"*70)
        
        step = max(1, len(positions) // 6)  # 6 точек
        for i in range(0, len(positions), step):
            if i < len(positions):
                p = positions[i]
                time_str = p['time'].strftime('%H:%M:%S')
                earth = p['shadow']
                sky = 100 - earth
                print(f"{time_str:<12} {p['lat']:<8.1f}° {p['lon']:<9.1f}° "
                      f"{p['distance_to_center']:<8.1f} {p['geometric_altitude']:<8.1f} "
                      f"{earth:<10.2f}% {sky:<10.2f}%")
        
        print("-"*70)
        
        # Экстремумы расстояния (апогей и перигей)
        max_dist_idx = np.argmax(distances)
        min_dist_idx = np.argmin(distances)
        
        print("\nЭКСТРЕМУМЫ РАССТОЯНИЯ ДО ЦЕНТРА ЗЕМЛИ:")
        print(f"АПОГЕЙ ({distances[max_dist_idx]:.1f} км, высота {geo_alts[max_dist_idx]:.1f} км):")
        print(f"  Время: {datetimes[max_dist_idx].strftime('%H:%M:%S UTC')}")
        print(f"  Широта: {lats[max_dist_idx]:.1f}°, Долгота: {lons[max_dist_idx]:.1f}°")
        print(f"  Затемнение: {shadows[max_dist_idx]:.2f}%")
        
        print(f"\nПЕРИГЕЙ ({distances[min_dist_idx]:.1f} км, высота {geo_alts[min_dist_idx]:.1f} км):")
        print(f"  Время: {datetimes[min_dist_idx].strftime('%H:%M:%S UTC')}")
        print(f"  Широта: {lats[min_dist_idx]:.1f}°, Долгота: {lons[min_dist_idx]:.1f}°")
        print(f"  Затемнение: {shadows[min_dist_idx]:.2f}%")
        
        # Размах орбиты
        dist_range = max(distances) - min(distances)
        alt_range = max(geo_alts) - min(geo_alts)
        print(f"\nРАЗМАХ ОРБИТЫ:")
        print(f"  Расстояние: {dist_range:.1f} км")
        print(f"  Высота: {alt_range:.1f} км")
        
        # Находим ближайшую точку к текущему времени
        time_diffs = [abs((dt - current_time).total_seconds()) for dt in datetimes]
        closest_idx = np.argmin(time_diffs)
        
        print(f"\nТЕКУЩЕЕ СОСТОЯНИЕ:")
        print(f"  Время: {current_time.strftime('%H:%M:%S UTC')}")
        print(f"  Расстояние до центра: {distances[closest_idx]:.1f} км")
        print(f"  Геометрическая высота: {geo_alts[closest_idx]:.1f} км")
        print(f"  Затемнение: {shadows[closest_idx]:.2f}%")
        print(f"  Видимость космоса: {100 - shadows[closest_idx]:.2f}%")
        
        print("\n" + "="*70)


def main():
    """Запуск расчета за последние 12 часов"""
    print("Запуск расчета затемнения неба для кубсата 239Alferov...")
    print("Период: последние 12 часов")
    print("Используется расстояние до центра Земли")
    print("Инициализация спутника...")
    
    # Создаем объект
    cube = CubeSatShadowSimple(64881)
    
    # Инициализируем спутник
    cube.initialize()
    
    # Печатаем сводку
    cube.print_summary()
    
    # Получаем данные
    data = cube.get_shadow_data()
    
    # Выводим статистику
    stats = data['statistics']
    metadata = data['metadata']
    current_utc = metadata['current_time'].strftime('%d.%m.%Y %H:%M:%S')
    
    print(f"\nТекущее время UTC: {current_utc}")
    print(f"Период анализа: {metadata['start_time'].strftime('%d.%m %H:%M')} - "
          f"{metadata['end_time'].strftime('%d.%m %H:%M')}")
    
    print("\nСТАТИСТИКА ДАННЫХ:")
    print(f"Затемнение: {stats['shadow']['mean']:.2f}% ± {stats['shadow']['std']:.2f}%")
    print(f"  Диапазон: {stats['shadow']['min']:.2f}% - {stats['shadow']['max']:.2f}%")
    print(f"Расстояние до центра: {stats['distance_to_center']['mean']:.1f} ± {stats['distance_to_center']['std']:.1f} км")
    print(f"  Диапазон: {stats['distance_to_center']['min']:.1f} - {stats['distance_to_center']['max']:.1f} км")
    print(f"Геометрическая высота: {stats['geometric_altitude']['mean']:.1f} ± {stats['geometric_altitude']['std']:.1f} км")
    print(f"  Диапазон: {stats['geometric_altitude']['min']:.1f} - {stats['geometric_altitude']['max']:.1f} км")
    
    # Строим график
    fig = cube.plot_shadow_evolution()
    
    # Показываем график
    plt.show()
    
    # Возвращаем данные для дальнейшего использования
    return data


if __name__ == "__main__":
    try:
        # Запускаем основную функцию
        shadow_data = main()
        
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем.")
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
        print("\nУстановите библиотеки: pip install numpy matplotlib skyfield requests")
