import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from skyfield.api import load, EarthSatellite
import requests
from datetime import timedelta

def get_tle_from_celestrak(norad_id=64881):
    """
    Получение TLE данных с Celestrak для спутника 239Alferov (NORAD ID: 64881)
    """
    celestrak_url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
    
    try:
        print(f"Получение TLE данных для NORAD ID {norad_id}...")
        response = requests.get(celestrak_url)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            if len(lines) >= 3:
                print("Данные успешно получены с Celestrak")
                return lines[0].strip(), lines[1].strip(), lines[2].strip()
            else:
                print("Недостаточно данных в ответе")
        else:
            print(f"Ошибка HTTP: {response.status_code}")
    except Exception as e:
        print(f"Ошибка получения данных: {e}")
    
    return None

def calculate_orbit_projection(satellite, duration_hours=12, steps=1000):
    """
    Расчет проекции орбиты спутника на поверхность Земли
    """
    ts = load.timescale()
    
    #Временной интервал для расчета - 12 часов
    start_time = ts.now()
    end_time = ts.utc(start_time.utc_datetime() + timedelta(hours=duration_hours))
    
    #Создание равномерно распределенных моментов времени
    times = ts.linspace(start_time, end_time, steps)
    
    #Инициализация списков для хранения данных
    longitudes = []
    latitudes = []
    
    #Расчет позиций для каждого момента времени
    for t in times:
        geocentric = satellite.at(t)
        subpoint = geocentric.subpoint()
        
        longitudes.append(subpoint.longitude.degrees)
        latitudes.append(subpoint.latitude.degrees)
    
    return longitudes, latitudes, start_time, end_time

def fix_longitude_discontinuity(longitudes, latitudes, threshold=180):
    """
    Разделяет траекторию на сегменты при пересечении антимеридиана
    """
    segments = []
    current_segment = []
    
    for i in range(len(longitudes)):
        if i > 0:
            #Проверяем резкий скачок долготы (переход через 180°)
            lon_diff = abs(longitudes[i] - longitudes[i-1])
            if lon_diff > threshold:
                #Сохраняем текущий сегмент и начинаем новый
                if current_segment:
                    segments.append(current_segment)
                current_segment = []
        
        current_segment.append((longitudes[i], latitudes[i]))
    
    #Добавляем последний сегмент
    if current_segment:
        segments.append(current_segment)
    
    return segments

def plot_orbit_on_mercator(longitudes, latitudes, start_time, end_time, satellite_name="239Alferov"):
    """
    Построение проекции орбиты на карту Mercator
    """
    #Создание фигуры
    plt.figure(figsize=(16, 12))
    
    #Первоначальный вариант карты
    m = Basemap(projection='merc', 
                llcrnrlat=-80, urcrnrlat=80,
                llcrnrlon=-180, urcrnrlon=180,
                lat_ts=20, resolution='c')
    
    m.drawcoastlines()
    m.fillcontinents(color='coral', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')
    
    #Разделяем траекторию на сегменты для устранения разрывов
    segments = fix_longitude_discontinuity(longitudes, latitudes)
    
    #Рисуем каждый сегмент отдельно
    for i, segment in enumerate(segments):
        seg_lons, seg_lats = zip(*segment)
        x, y = m(seg_lons, seg_lats)
        m.plot(x, y, 'red', linewidth=2.0, 
               label=f'Орбита {satellite_name}' if i == 0 else "")
    
    #Отметка начальной позиции
    x_start, y_start = m([longitudes[0]], [latitudes[0]])
    m.plot(x_start, y_start, 'bo', markersize=10, markeredgecolor='black', markeredgewidth=1.5, 
           label=f'Старт: {start_time.utc_strftime("%d.%m %H:%M UTC")}')
    
    #Отметка конечной позиции
    x_end, y_end = m([longitudes[-1]], [latitudes[-1]])
    m.plot(x_end, y_end, 'go', markersize=10, markeredgecolor='black', markeredgewidth=1.5, 
           label=f'Финиш: {end_time.utc_strftime("%d.%m %H:%M UTC")}')
    
    #Настройка отображения
    duration_hours = (end_time - start_time) * 24
    orbital_period = 24 / 15.2  #приблизительный орбитальный период в часах (из среднего движения)
    orbits_count = duration_hours / orbital_period
    
    plt.title(f"Проекция орбиты {satellite_name} (NORAD 64881)\n"
              f"Продолжительность: {duration_hours:.0f} часов | Точек: {len(longitudes)} | Витков: ~{orbits_count:.1f}", 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.legend(loc='upper right', fontsize=12)
    
    return plt

def main():
    """
    Основная функция для построения проекции орбиты
    """
    print("Построение проекции орбиты для спутника 239Alferov (NORAD ID: 64881)")
    print("=" * 60)
    
    #Получение TLE данных
    tle_data = get_tle_from_celestrak(64881)
    
    if tle_data is None:
        print("Используются резервные TLE данные...")
        name, line1, line2 = get_backup_tle()
    else:
        name, line1, line2 = tle_data
    
    print(f"\nСпутник: {name}")
    print(f"TLE строка 1: {line1}")
    print(f"TLE строка 2: {line2}")
    
    #Создание объекта спутника
    ts = load.timescale()
    satellite = EarthSatellite(line1, line2, name, ts)
    
    #Расчет орбитальной траектории на 12 часов с 1000 точками
    print("\nРасчет орбитальной траектории на 12 часов...")
    longitudes, latitudes, start_time, end_time = calculate_orbit_projection(satellite, duration_hours=12, steps=1000)
    
    #Вывод информации о траектории
    print(f"Рассчитано {len(longitudes)} точек траектории")
    print(f"Диапазон широт: {min(latitudes):.2f}° до {max(latitudes):.2f}°")
    print(f"Диапазон долгот: {min(longitudes):.2f}° до {max(longitudes):.2f}°")
    print(f"Время начала: {start_time.utc_strftime('%d.%m.%Y %H:%M UTC')}")
    print(f"Время окончания: {end_time.utc_strftime('%d.%m.%Y %H:%M UTC')}")
    
    #Расчет примерного количества витков
    orbital_period = 24 / 15.2  #часов на виток
    orbits_count = 12 / orbital_period
    print(f"Примерное количество витков за 12 часов: {orbits_count:.1f}")
    
    
    #Построение карты
    print("Построение карты...")
    plot = plot_orbit_on_mercator(longitudes, latitudes, start_time, end_time, name)
    
    #Отображение карты
    plt.show()
    
    #Сохранение результата
    try:
        plt.gcf().savefig('239alferov_12h_orbit_projection.png', dpi=300, bbox_inches='tight')
        print("\nРезультат сохранен: 239alferov_12h_orbit_projection.png")
    except:
        print("\nНе удалось сохранить файл, но карта отображается на экране")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ошибка выполнения: {e}")
        print("Проверьте установку библиотек: pip install skyfield matplotlib basemap requests numpy")
