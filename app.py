import streamlit as st
import pandas as pd
import plotly.io as pio
pio.renderers.default = 'svg'
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import re

from core import (
    FarmModel, FieldAgent, Crop, Soil, Operation, TechCard,
    TaskType, MachineType, CropPhase, CropPhaseInfo,
    run_simulation_with_structure, optimize_structure, compute_yield_scenarios,
    compute_period_norms, adaptive_resource_optimization, compute_full_variable_cost,
    plot_gdd_and_phases, plot_daily_stress_and_yield
)
from viz import (
    plot_gantt_by_operations, plot_soil_moisture, plot_machine_utilization,
    plot_weather,
    display_detailed_adaptation_table,
    plot_yield_by_scenario,
    plot_rejection_reasons, plot_downtime_events
)

st.set_page_config(
    page_title="Адаптивная цифровая модель агропредприятия Крыма",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _flatten_agent_df(agent_df):
    if agent_df is None or agent_df.empty:
        return pd.DataFrame()
    if isinstance(agent_df.columns, pd.MultiIndex):
        flat = agent_df.stack().reset_index()
        flat.columns = ['Step', 'AgentID', 'Variable', 'Value']
        flat = flat.pivot(index=['Step', 'AgentID'], columns='Variable', values='Value').reset_index()
    else:
        flat = agent_df.reset_index()
        if 'Step' not in flat.columns:
            flat = flat.rename(columns={'index': 'Step'})
        if 'AgentID' not in flat.columns:
            if 'level_1' in flat.columns:
                flat = flat.rename(columns={'level_1': 'AgentID'})
            else:
                flat['AgentID'] = flat.index
    return flat

data_configs = {
    "crops": {
        "title": "Культуры (crops.csv)",
        "session_key": "crops_df",
        "file_name": "crops.csv",
        "int_columns": ["crop_id", "T_base", "T_min", "T_opt1", "T_opt2", "T_max", "yield_pot", "price", "seed_cost",
                        "N_need", "P_need", "K_need", "water_need", "irrigation_threshold",
                        "irrigation_amount", "sowing_start", "sowing_end", "harvest_start", "harvest_end",
                        "establishment_cost", "useful_life_years"] 
    },
    "phases": {
        "title": "Фазы развития (crop_phases.csv)",
        "session_key": "phases_df",
        "file_name": "crop_phases.csv",
        "int_columns": ["phase_id", "crop_id", "gdd_required", "sensitivity", "Tmin", "Topt1", "Topt2", "Tmax",
                        "P_min", "P_opt1", "P_opt2", "P_max", "PAR_min", "PAR_opt1", "PAR_opt2", "PAR_max",
                        "weight_temp", "weight_water", "weight_par"]
    },
    "soils": {
        "title": "Почвы (soils.csv)",
        "session_key": "soils_df",
        "file_name": "soils.csv",
        "int_columns": ["soil_id", "field_capacity", "wilting_point", "infiltration_rate", "organic_matter"]
    },
    "fields": {
        "title": "Поля (fields.csv)",
        "session_key": "fields_df",
        "file_name": "fields.csv",
        "int_columns": ["field_id", "soil_id", "area", "initial_moisture", "initial_N", "initial_P", "initial_K", "cadastral_value"]   # NEW
    },
    "machines": {
        "title": "Техника (machines.csv)",
        "session_key": "machines_df",
        "file_name": "machines.csv",
        "int_columns": ["machine_id", "productivity", "fuel_consumption", "fuel_capacity", "cost_hour", "width", "speed", "initial_cost", "useful_life_years"]   # NEW
    },
    "workers": {
        "title": "Рабочие (workers.csv)",
        "session_key": "workers_df",
        "file_name": "workers.csv",
        "int_columns": ["worker_id", "cost_per_hour", "max_hours_per_day"]
    },
    "storages": {
        "title": "Хранилища (storages.csv)",
        "session_key": "storages_df",
        "file_name": "storages.csv",
        "int_columns": ["storage_id", "capacity", "x", "y"],
        "float_columns": ["cost_per_ton_per_day", "annual_loss_rate"]   # NEW
    },
    "operations": {
        "title": "Операции (operations.csv)",
        "session_key": "operations_df",
        "file_name": "operations.csv",
        "int_columns": ["operation_id", "duration_per_ha", "fuel_per_ha", "material_consumption", "max_soil_moisture"]
    },
    "tech_cards": {
        "title": "Технологические карты (tech_cards.csv)",
        "session_key": "tech_cards_df",
        "file_name": "tech_cards.csv",
        "int_columns": ["tech_card_id", "crop_id", "operation_id", "day_start", "day_end", "priority", "amount_per_ha"]
    },
    "weather": {
        "title": "Погодные данные (weather_daily.csv)",
        "session_key": "weather_df",
        "file_name": "weather_daily.csv",
        "int_columns": ["scenario_id", "day"],
        "float_columns": ["tmin", "tmax", "precip", "par"]
    },
    "scenarios": {
        "title": "Погодные сценарии (weather_scenarios.csv)",
        "session_key": "weather_scenarios_df",
        "file_name": "weather_scenarios.csv",
        "int_columns": ["scenario_id"],
        "float_columns": ["probability"]
    },
    "constraints": {
        "title": "Ограничения (constraints.csv)",
        "session_key": "constraints_df",
        "file_name": "constraints.csv",
        "int_columns": ["constraint_id", "value"]
    },
    "economic_params": {   
        "title": "Экономические параметры (economic_params.csv)",
        "session_key": "economic_params_df",
        "file_name": "economic_params.csv",
        "int_columns": [],
        "float_columns": ["param_value"]
    },
    "prices": {
        "title": "Цены на ресурсы (prices.csv)",
        "session_key": "prices_df",
        "file_name": "prices.csv",
        "int_columns": ["resource_id"],
        "float_columns": ["price"]
    },
    "risks": {
    "title": "Риски (risks.csv)",
    "session_key": "risks_df",
    "file_name": "risks.csv",
    "int_columns": ["risk_id", "downtime_days"],
    "float_columns": ["penalty_yield", "fixed_cost_rub", "probability"]
    }
}

def fix_dtypes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if "int_columns" in config:
        for col in config["int_columns"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    if "float_columns" in config:
        for col in config["float_columns"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df

def render_data_tab(tab, config: dict):
    with tab:
        st.subheader(config["title"])
        edited_df = st.data_editor(
            st.session_state[config["session_key"]],
            num_rows="dynamic",
            key=f"{config['session_key']}_editor"
        )
        if st.button("Сохранить изменения", key=f"save_{config['session_key']}"):
            edited_df = fix_dtypes(edited_df, config)
            st.session_state[config["session_key"]] = edited_df
            st.success("Сохранено!")
            st.rerun()
        uploaded = st.file_uploader(
            f"Загрузить {config['file_name']}",
            type='csv',
            key=f"{config['session_key']}_upload"
        )
        if uploaded:
            df = pd.read_csv(uploaded)
            df = fix_dtypes(df, config)
            st.session_state[config["session_key"]] = df
            st.rerun()

def init_dataframes():
    if 'crops_df' not in st.session_state:
        st.session_state.crops_df = pd.DataFrame({
            'crop_id': [1, 2, 3],
            'name': ['Виноград Каберне Совиньон', 'Виноград Мерло', 'Лаванда узколистная'],
            'type': ['vine', 'vine', 'oilseed'],
            'T_base': [10, 8, 5],
            'T_min': [10, 8, 5],
            'T_opt1': [20, 18, 15],
            'T_opt2': [30, 28, 25],
            'T_max': [40, 38, 35],
            'yield_pot': [14.0, 17.0, 3.5],
            'price': [45000, 35000, 120000],
            'seed_cost': [0, 0, 8000],
            'N_need': [120, 100, 80],
            'P_need': [80, 70, 50],
            'K_need': [70, 60, 55],
            'water_need': [4200, 3800, 3200],
            'irrigation_threshold': [200, 200, 200],
            'irrigation_amount': [40, 35, 30],
            'max_share': [0.5, 0.5, 0.3],
            'sowing_start': [90, 85, 80],
            'sowing_end': [130, 125, 100],
            'harvest_start': [260, 250, 180],
            'harvest_end': [290, 280, 200],
            'establishment_cost': [200000, 180000, 150000],   # руб/га
            'useful_life_years': [25, 25, 15]
        })

    if 'phases_df' not in st.session_state:
        st.session_state.phases_df = pd.DataFrame({
            'phase_id': list(range(1, 19)),
            'crop_id': [1,1,1,1,1,1,  2,2,2,2,2,2,  3,3,3,3,3,3],
            'phase_name': [
                'Распускание почек','Цветение','Завязывание ягод','Веразон','Созревание','Сбор',
                'Распускание почек','Цветение','Завязывание ягод','Веразон','Созревание','Сбор',
                'Вегетация','Бутонизация','Цветение','Завязывание','Созревание семян','Увядание'
            ],
            'phase_enum': [
                'BUD_BREAK','FLOWERING','BERRY_GROWTH','VERAISON','RIPENING','MATURE',
                'BUD_BREAK','FLOWERING','BERRY_GROWTH','VERAISON','RIPENING','MATURE',
                'VEGETATIVE','BUDDING','FLOWERING','BERRY_GROWTH','SEED_MATURITY','SENESCENCE'
            ],
            'gdd_required': [80,250,400,800,1200,1500, 100,300,500,900,1500,1649, 100,250,500,700,1100,1300],
            'kc': [0.5,0.8,1.05,1.1,0.9,0.9, 0.55,0.85,1.1,1.15,0.95,0.9, 0.4,0.7,0.9,0.95,0.8,0.5],
            'sensitivity': [0.6,0.8,0.7,0.6,0.5,0.3, 0.7,0.9,0.8,0.7,0.6,0.3, 0.5,0.7,0.8,0.7,0.6,0.4],
            'next_phase': [
                'FLOWERING','BERRY_GROWTH','VERAISON','RIPENING','MATURE',None,
                'FLOWERING','BERRY_GROWTH','VERAISON','RIPENING','MATURE',None,
                'BUDDING','FLOWERING','BERRY_GROWTH','SEED_MATURITY','SENESCENCE',None
            ],
            'Tmin': [8,12,12,15,15,15, 8,12,12,15,15,15, 5,8,12,12,10,5],
            'Topt1': [15,18,18,20,20,20, 16,20,20,22,22,20, 10,15,18,18,15,10],
            'Topt2': [22,26,28,30,30,28, 24,30,32,34,34,27, 20,25,28,28,29,30],
            'Tmax': [30,34,36,38,38,40, 32,36,38,40,40,50, 28,32,35,35,35,35],
            'P_min': [0.0001,0,0,0.00001,0,-1, 0.0001,0.0002,0.0002,1,0,-1, 0.0005,1,2,1.5,0.5,0],
            'P_opt1': [0.01,0.001,0.1,0.01,0.01,0.0001, 0.1,0.5,0.6,0,1,0.001, 2,3,4,3,2,1],
            'P_opt2': [5,6,8,5,4,4, 6,8,10,6,4,8, 4,5,6,5,4,2],
            'P_max': [10,12,15,10,8,7, 12,15,20,12,8,8, 8,10,12,10,8,5],
            'PAR_min': [5,8,10,12,10,7, 5,8,10,12,10,7, 3,5,8,8,5,3],
            'PAR_opt1': [10,15,18,20,18,18, 10,15,18,22,18,18, 8,12,15,15,12,8],
            'PAR_opt2': [20,25,28,30,28,28, 22,28,32,34,30,30, 18,22,25,25,20,15],
            'PAR_max': [30,35,38,40,38,38, 32,38,42,45,40,40, 25,30,35,35,28,20],
            'weight_temp': [0.5,0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5,0.5,0.5],
            'weight_water': [0.3,0.3,0.3,0.3,0.3,0.3, 0.3,0.3,0.3,0.3,0.3,0.3, 0.3,0.3,0.3,0.3,0.3,0.3],
            'weight_par': [0.2,0.2,0.2,0.2,0.2,0.2, 0.2,0.2,0.2,0.2,0.2,0.2, 0.2,0.2,0.2,0.2,0.2,0.2]
        })
    
    if 'soils_df' not in st.session_state:
        st.session_state.soils_df = pd.DataFrame({
            'soil_id': [1],
            'name': ['Чернозём выщелоченный'],
            'field_capacity': [350],
            'wilting_point': [150],
            'infiltration_rate': [30],
            'bulk_density': [1.25],
            'organic_matter': [5.2],
            'pH': [6.5]
        })
    
    if 'fields_df' not in st.session_state:
        st.session_state.fields_df = pd.DataFrame({
            'field_id': [1,2,3],
            'area': [50,40,60],
            'soil_id': [1,1,1],
            'crop_plan': ['Виноград Каберне Совиньон', 'Виноград Мерло', 'Лаванда узколистная'],
            'x': [0,100,20],
            'y': [0,50,80],
            'initial_moisture': [160,150,155],
            'initial_N': [65,60,70],
            'initial_P': [45,40,50],
            'initial_K': [55,50,60],
            'cadastral_value': [300000, 300000, 300000]   # руб/га
        })
    
    if 'machines_df' not in st.session_state:
        st.session_state.machines_df = pd.DataFrame({
            'machine_id': [1,2,3,4,5],
            'type': ['tractor','harvester','irrigator','sprayer','transporter'],
            'brand': ['John Deere 6130','New Holland Braud','Valley 8000','Amazone UX','KAMAZ'],
            'productivity': [12,5,10,20,30],
            'fuel_consumption': [12,20,8,10,25],
            'fuel_capacity': [300,400,200,300,400],
            'cost_hour': [2500,4000,1500,2000,3000],
            'width': [3,2,50,24,None],
            'speed': [10,5,2,12,None],
            'requires_operator': [True,True,True,True,True],
            'initial_cost': [2500000, 8000000, 1500000, 3000000, 3500000],
            'useful_life_years': [10, 10, 12, 8, 12]
        })
    
    if 'workers_df' not in st.session_state:
        st.session_state.workers_df = pd.DataFrame({
            'worker_id': [1,2,3],
            'name': ['Иван Петров','Сергей Иванов','Мария Иванова'],
            'qualification': ['tractor_driver','harvester_operator','general'],
            'cost_per_hour': [350,400,280],
            'max_hours_per_day': [10,10,8]
        })
    
    if 'storages_df' not in st.session_state:
        st.session_state.storages_df = pd.DataFrame({
            'storage_id': [1,2,3],
            'type': ['grain','fertilizer','fuel'],
            'capacity': [1200,300,200],
            'x': [10,5,2],
            'y': [10,15,2],
            'cost_per_ton_per_day': [5.0, 1.0, 0.5],
            'annual_loss_rate': [0.02, 0.01, 0.005]
        })
    
    if 'operations_df' not in st.session_state:
        st.session_state.operations_df = pd.DataFrame({
            'operation_id': [1,2,3,4,5,6,7,8],
            'name': ['Вспашка','Культивация','Посадка','Полив','Опрыскивание','Уборка винограда','Уборка лаванды','Транспортировка'],
            'task_type': ['plow','cultivate','sow','irrigate','spray','harvest','harvest','transport'],
            'machine_type': ['tractor','tractor','tractor','irrigator','sprayer','harvester','harvester','transporter'],
            'duration_per_ha': [0.5,0.3,0.8,0.2,0.2,1.2,1.0,0.1],
            'fuel_per_ha': [18,10,12,5,6,22,18,8],
            'material_consumption': [0,0,0,40,2,0,0,0],
            'material_type': [None,None,'seed','water','pesticide',None,None,None],
            'max_soil_moisture': [9999,9999,9999,320,9999,9999,9999,9999],
            'requires_operator': [True,True,True,True,True,True,True,True]
        })
    
    if 'tech_cards_df' not in st.session_state:
        st.session_state.tech_cards_df = pd.DataFrame({
            'tech_card_id': list(range(1,19)),
            'crop_id': [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3],
            'phase_enum': ['BUD_BREAK','BUD_BREAK','BUD_BREAK','FLOWERING','BERRY_GROWTH','RIPENING',
                           'BUD_BREAK','BUD_BREAK','BUD_BREAK','FLOWERING','BERRY_GROWTH','RIPENING',
                           'VEGETATIVE','VEGETATIVE','VEGETATIVE','BUDDING','FLOWERING','FLOWERING'],
            'operation_id': [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,7,8],
            'day_start': [0,4,5,0,0,5,0,2,5,0,0,5,0,3,2,0,3,1],
            'day_end':   [3,8,10,5,5,10,7,7,15,4,8,5,5,8,7,3,3,2],
            'priority': [10,10,20,15,10,20,10,10,20,15,10,20,10,10,20,15,20,10],
            'amount_per_ha': [0,0,2500,40,3,0,0,0,2200,35,2.5,0,0,0,2,30,0,0]
        })
    
    if 'weather_df' not in st.session_state:
        np.random.seed(42)
        days = np.arange(1, 366)
        weather_list = []
        base_temp = 14.0
        amplitude = 14.0
        peak_day = 196
        for sid, name, temp_offset, factor_precip in [(1, 'Нормальный', 0, 1.0),
                                                       (2, 'Сухой', 2, 0.6),
                                                       (3, 'Влажный', 0, 1.4)]:
            tmean_base = base_temp + amplitude * np.sin(2 * np.pi * (days - 105) / 365)
            tmean_base = tmean_base + temp_offset
            tmin = tmean_base - 6.0
            tmax = tmean_base + 6.0
            if name == 'Влажный':
                precip = np.where(np.random.random(365) < 0.2,
                                  np.random.gamma(2.0, 8, 365),
                                  np.random.gamma(0.8, 2, 365)) * 1.5
            elif name == 'Сухой':
                precip = np.random.gamma(0.6, 1.5, 365) * 0.6
            else:
                precip = np.where(np.random.random(365) < 0.1,
                                  np.random.gamma(1.5, 5, 365),
                                  np.random.gamma(0.7, 1.8, 365))
            precip = np.maximum(0, precip)
            par = 12 + 18 * np.sin(2 * np.pi * (days - 80) / 365)
            par = np.clip(par, 6, 30)
            for d in days:
                weather_list.append({
                    'scenario_id': sid,
                    'day': d,
                    'tmin': round(tmin[d-1], 1),
                    'tmax': round(tmax[d-1], 1),
                    'precip': round(precip[d-1], 1),
                    'par': round(par[d-1], 1)
                })
        st.session_state.weather_df = pd.DataFrame(weather_list)
    
    if 'weather_scenarios_df' not in st.session_state:
        st.session_state.weather_scenarios_df = pd.DataFrame({
            'scenario_id': [1,2,3],
            'name': ['Нормальный', 'Сухой', 'Влажный'],
            'probability': [0.6, 0.2, 0.2]
        })
    if 'constraints_df' not in st.session_state:
        st.session_state.constraints_df = pd.DataFrame({
            'constraint_id': [1,2,3,4],
            'type': ['water','budget','labor','machine_hours'],
            'value': [80000, 5000000, 2000, 1000],
            'unit': ['м³','руб','часы','часы'],
            'description': ['Общий лимит воды на сезон','Бюджет на операционные расходы',
                            'Максимум человеко-часов в месяц','Лимит работы техники в месяц']
        })
    
    if 'market_prices_df' not in st.session_state:
        st.session_state.market_prices_df = pd.DataFrame({
            'price_id': [1,2,3,4,5,6,7,8,9],
            'crop_id': [1,1,1,2,2,2,3,3,3],
            'month': [9,10,11,9,10,11,7,8,9],
            'price': [48000,52000,50000,37000,40000,38000,130000,125000,120000]
        })

    if 'prices_df' not in st.session_state:
        st.session_state.prices_df = pd.DataFrame({
        'resource_id': [1,2,3,4,5],
        'resource_name': ['fuel', 'water', 'seed', 'fertilizer', 'pesticide'],
        'price': [60, 10, 50, 100, 500],
        'unit': ['руб/л', 'руб/м³', 'руб/кг', 'руб/кг', 'руб/кг']
        })

    if 'economic_params_df' not in st.session_state:
        st.session_state.economic_params_df = pd.DataFrame({
        'param_name': ['depreciation_rate_machines', 'depreciation_rate_perennials',
                       'property_tax_rate', 'profit_tax_rate', 'admin_cost_fixed',
                       'storage_cost_per_ton_per_day', 'annual_storage_loss_rate',
                       'realization_rate'],
        'param_value': [0.10, 0.05, 0.022, 0.20, 500000.0, 5.0, 0.02, 0.95],
        'unit': ['%', '%', '%', '%', 'руб', 'руб/т/день', '%', '%'],
        'description': ['Годовая норма амортизации техники', 
                        'Годовая норма амортизации многолетних культур',
                        'Ставка налога на имущество', 'Ставка налога на прибыль',
                        'Фиксированные административные расходы в год',
                        'Стоимость хранения 1 тонны продукции в день', 
                        'Годовой процент потерь при хранении',
                        'Доля реализованного урожая (0..1)']
        })
    if 'risks_df' not in st.session_state:
        st.session_state.risks_df = pd.DataFrame({
        'risk_id': [1, 2, 3, 4, 5],
        'name': ['Весенние заморозки', 'Милдью', 'Скособр', 'Поломка техники', 'Воровство'],
        'crop_type': ['vine', 'vine', 'vine', 'all', 'all'],
        'phenophase': ['BUD_BREAK', 'FLOWERING', 'BUD_BREAK', 'any', 'HARVEST'],
        'trigger_type': ['temp_below_days', 'rain_after_temp_days', 'random', 'random', 'random'],
        'trigger_value': ['-3,2', '10,10,2', '0.2', '0.1', '0.15'],
        'penalty_yield': [0.6, 0.3, 0.25, 0.0, 0.1],
        'fixed_cost_rub': [0, 0, 0, 50000, 0],
        'downtime_days': [0, 0, 0, 3, 0],
        'probability': [1.0, 0.6, 0.0001, 0.005, 0.2],
        })

init_dataframes()

def build_crops_from_dataframes(crops_df: pd.DataFrame, phases_df: pd.DataFrame) -> Dict[str, Crop]:
    from core import Crop, CropPhaseInfo
    phases_by_crop = {}
    for _, row in phases_df.iterrows():
        crop_id = row['crop_id']
        phase_enum = row['phase_enum']
        weights = {'temp': row.get('weight_temp', 0.5),
                   'water': row.get('weight_water', 0.3),
                   'par': row.get('weight_par', 0.2)}
        phase_info = CropPhaseInfo(
            phase_enum=phase_enum,
            gdd_required=row['gdd_required'],
            kc=row['kc'],
            sensitivity=row['sensitivity'],
            next_phase=row.get('next_phase'),
            weights=weights,
            Tmin=row.get('Tmin'), Topt1=row.get('Topt1'), Topt2=row.get('Topt2'), Tmax=row.get('Tmax'),
            P_min=row.get('P_min'), P_opt1=row.get('P_opt1'), P_opt2=row.get('P_opt2'), P_max=row.get('P_max'),
            PAR_min=row.get('PAR_min'), PAR_opt1=row.get('PAR_opt1'), PAR_opt2=row.get('PAR_opt2'), PAR_max=row.get('PAR_max')
        )
        phases_by_crop.setdefault(crop_id, {})[phase_enum] = phase_info
    crops = {}
    for _, row in crops_df.iterrows():
        crop_id = row['crop_id']
        phases = phases_by_crop.get(crop_id, {})
        crop = Crop(row.to_dict(), phases)
        crops[crop.name] = crop
    return crops

def build_soils_dict(soils_df: pd.DataFrame) -> Dict[str, Soil]:
    soils = {}
    for _, row in soils_df.iterrows():
        soils[row['name']] = Soil(row.to_dict())
    return soils

def build_fields_list(fields_df: pd.DataFrame, soils_dict: Dict[str, Soil]) -> List[dict]:
    fields = []
    for _, row in fields_df.iterrows():
        soil_id = row['soil_id']
        soil_name = None
        for name, soil in soils_dict.items():
            if soil.soil_id == soil_id:
                soil_name = name
                break
        if soil_name is None:
            soil_name = list(soils_dict.keys())[0]
        fields.append({
            'field_id': row['field_id'],
            'area': row['area'],
            'soil_name': soil_name,
            'crop_plan': row.get('crop_plan'),
            'x': row.get('x', 0),
            'y': row.get('y', 0),
            'initial_moisture': row.get('initial_moisture', 160),
            'initial_N': row.get('initial_N', 50),
            'initial_P': row.get('initial_P', 30),
            'initial_K': row.get('initial_K', 40),
            'cadastral_value': row.get('cadastral_value', 0.0)
        })
    return fields

def build_machines_list(machines_df: pd.DataFrame) -> List[dict]:
    return machines_df.to_dict('records')

def build_workers_list(workers_df: pd.DataFrame) -> List[dict]:
    return workers_df.to_dict('records')

def build_storages_list(storages_df: pd.DataFrame) -> List[dict]:
    return storages_df.to_dict('records')

def build_operations_dict(operations_df: pd.DataFrame) -> Dict[int, Operation]:
    ops = {}
    for _, row in operations_df.iterrows():
        ops[row['operation_id']] = Operation(row.to_dict())
    return ops

def build_tech_cards_list(tech_cards_df: pd.DataFrame) -> List[TechCard]:
    return [TechCard(row.to_dict()) for _, row in tech_cards_df.iterrows()]

def get_planned_dates(crop_name: str, tech_card: pd.Series, crops_df: pd.DataFrame, sowing_start_col='sowing_start') -> Tuple[int, int]:
    crop_row = crops_df[crops_df['name'] == crop_name].iloc[0]
    phase_start_abs = crop_row[sowing_start_col]
    start_day = phase_start_abs + tech_card['day_start']
    end_day = phase_start_abs + tech_card['day_end']
    return start_day, end_day

if 'optimize_flag_prev' not in st.session_state:
    st.session_state.optimize_flag_prev = False

with st.sidebar:
    st.header("🌾 Управление")
    mode = st.radio("Режим", ["📊 Модель", "📁 Данные"], index=0)

    if mode == "📊 Модель":
        with st.form(key="simulation_form"):
            st.subheader("⚙️ Параметры")
            optimize_flag = st.checkbox("Оптимизировать структуру площадей (LP)", value=True)

            if optimize_flag != st.session_state.optimize_flag_prev:
                for key in ['robust', 'lp_result', 'lp_profit', 'current_results', 'current_results_no_adapt']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.optimize_flag_prev = optimize_flag

            available_crops = list(st.session_state.crops_df['name'])
            selected_crops = st.multiselect("Культуры-кандидаты", available_crops, default=available_crops) if optimize_flag else []

            scenarios_df = st.session_state.weather_scenarios_df
            if scenarios_df.empty:
                st.warning("Нет погодных сценариев. Добавьте их во вкладке 'Данные'.")
                use_scenario = {}
                scenario_probs = {}
                selected_scenarios = []
            else:
                st.subheader("Погодные сценарии")
                use_scenario = {}
                prob = {}
                for _, row in scenarios_df.iterrows():
                    scen_id = row['scenario_id']
                    scen_name = row['name']
                    use_scenario[scen_name] = st.checkbox(scen_name, value=True, key=f"use_scen_{scen_id}")
                    prob[scen_name] = st.slider(f"Вероятность ({scen_name})", 0.0, 1.0,
                                                 float(row.get('probability', 1.0/len(scenarios_df))),
                                                 key=f"prob_scen_{scen_id}")
                selected_scenarios = [s for s in use_scenario if use_scenario[s]]
                if not selected_scenarios:
                    st.warning("Выберите хотя бы один сценарий")

                total_prob = sum(prob[s] for s in selected_scenarios)
                if total_prob > 0:
                    scenario_probs = {s: prob[s]/total_prob for s in selected_scenarios}
                else:
                    scenario_probs = {s: 1.0/len(selected_scenarios) for s in selected_scenarios}

            total_area = st.number_input("Общая площадь (га)", value=float(st.session_state.fields_df['area'].sum()))
            water_limit = st.number_input("Лимит воды (м³)", value=500000)
            budget_limit = st.number_input("Бюджет (руб)", value=5_000_000)
            days = st.number_input("Дней симуляции", min_value=1, max_value=1000, value=365)
            compare_plan = st.checkbox("Сравнить с текущим планом (исходная структура)", value=True)
            submitted = st.form_submit_button("Запустить модель", type="primary", use_container_width=True)

        if 'robust' in st.session_state:
            st.subheader("📌 Выбор сценария для просмотра")
            scenario_options = ["Общий робастный план"] + list(st.session_state.robust['scenario_results'].keys())
            selected_view_scenario = st.selectbox("Сценарий", scenario_options, key="view_scenario")
            st.session_state.selected_view_scenario = selected_view_scenario

if mode == "📊 Модель" and submitted:
    with st.spinner("Подготовка данных..."):
        crops_dict = build_crops_from_dataframes(st.session_state.crops_df, st.session_state.phases_df)
        if optimize_flag:
            crops_dict = {name: crops_dict[name] for name in selected_crops if name in crops_dict}
        st.session_state.crops_dict = crops_dict

        soils_dict = build_soils_dict(st.session_state.soils_df)
        fields = build_fields_list(st.session_state.fields_df, soils_dict)
        machines = build_machines_list(st.session_state.machines_df)
        workers = build_workers_list(st.session_state.workers_df)
        storages = build_storages_list(st.session_state.storages_df)
        operations = build_operations_dict(st.session_state.operations_df)
        tech_cards = build_tech_cards_list(st.session_state.tech_cards_df)
        st.session_state.operations_dict = operations

        weather_df = st.session_state.weather_df
        scenario_name_to_id = {row['name']: row['scenario_id'] for _, row in st.session_state.weather_scenarios_df.iterrows()}
        weather_by_name = {}
        for scen in selected_scenarios:
            sid = scenario_name_to_id[scen]
            weather_by_name[scen] = weather_df[weather_df['scenario_id'] == sid].copy()
        st.session_state.weather_by_name = weather_by_name

        economic_params = {}
        for _, row in st.session_state.economic_params_df.iterrows():
            economic_params[row['param_name']] = row['param_value']

        lp_result = None
        lp_profit_val = 0.0
        if optimize_flag and selected_scenarios:
            yield_scenarios, w = compute_yield_scenarios(crops_dict, weather_by_name)
            var_costs = {}
            resource_prices = {}
            labor_cost_per_hour = 300
            if not st.session_state.prices_df.empty:
                for _, row in st.session_state.prices_df.iterrows():
                    resource_prices[row['resource_name']] = row['price']
            else:
                resource_prices = {'fuel': 60, 'water': 10, 'seed': 50, 'fertilizer': 100, 'pesticide': 500}
                
            for crop_name, crop in crops_dict.items():
                var_costs[crop_name] = compute_full_variable_cost(
                    crop, operations, tech_cards,
                    labor_cost_per_hour, resource_prices
                )
            base_scenario = selected_scenarios[0]
            base_weather_df = weather_by_name[base_scenario]
            labor_norm, machine_norm = compute_period_norms(
                crops_dict, st.session_state.tech_cards_df, st.session_state.operations_df,
                base_weather_df, num_periods=36
            )
            periods = list(range(36))
            labor_limits = {(p, 'general'): 2000 for p in periods}
            machine_limits = {}
            for p in periods:
                machine_limits[(p, 'tractor')] = 1500
                machine_limits[(p, 'harvester')] = 300
                machine_limits[(p, 'irrigator')] = 400
                machine_limits[(p, 'sprayer')] = 200
                machine_limits[(p, 'transporter')] = 500
            min_areas = {c: 0.0 for c in crops_dict}
            max_shares = {c: crops_dict[c].max_share for c in crops_dict}

            try:
                lp_result, lp_profit_val = optimize_structure(
                    crops=crops_dict,
                    total_area=total_area,
                    water_limit=water_limit,
                    budget_limit=budget_limit,
                    scenario_probs=scenario_probs,
                    yield_scenarios=yield_scenarios,
                    var_costs=var_costs,
                    labor_norm=labor_norm,
                    labor_limits=labor_limits,
                    machine_norm=machine_norm,
                    machine_limits=machine_limits,
                    min_areas=min_areas,
                    max_shares=max_shares,
                    periods=periods
                )
                st.success("LP решена успешно")
            except Exception as e:
                st.error(f"Ошибка LP: {e}")
                lp_result = None
                lp_profit_val = 0.0
        else:
            optimize_flag = False
            lp_result = None
            lp_profit_val = 0.0

        if compare_plan:
            current_areas = {}
            for f in fields:
                crop = f.get('crop_plan')
                if crop:
                    current_areas[crop] = current_areas.get(crop, 0) + f['area']
            st.session_state.current_areas = current_areas
        else:
            st.session_state.current_areas = None

    with st.spinner("Запуск адаптивной оптимизации по всем сценариям..."):
        try:
            robust = adaptive_resource_optimization(
                crops=crops_dict,
                total_area=total_area,
                initial_water_limit=water_limit,
                initial_budget=budget_limit,
                initial_labor_limits=labor_limits if optimize_flag else {},
                initial_machine_limits=machine_limits if optimize_flag else {},
                yield_scenarios=yield_scenarios if optimize_flag else None,
                scenario_probs=scenario_probs if optimize_flag else None,
                min_areas=min_areas if optimize_flag else {},
                max_shares=max_shares if optimize_flag else {},
                fields_data=fields,
                soils=soils_dict,
                machines_data=machines,
                workers_data=workers,
                storages_data=storages,
                operations=operations,
                tech_cards=tech_cards,
                weather_daily_by_scenario=weather_by_name,
                var_costs=var_costs if optimize_flag else None,
                optimal_areas=lp_result,
                constant_costs=0.0,
                resource_prices=resource_prices,
                economic_params=economic_params,
                risks_df=st.session_state.risks_df,
            )
            st.session_state.robust = robust
            st.session_state.phenology_warnings = robust.get('phenology_warnings', {})
            st.success("Робастный план рассчитан!")
        except Exception as e:
            st.error(f"Ошибка: {e}")
            st.stop()

    with st.spinner("Запуск имитации текущей структуры (с адаптацией)..."):
        current_results = {}
        original_fields = build_fields_list(st.session_state.fields_df, soils_dict)
        for scen_name, weather_df in weather_by_name.items():
            weather_daily = weather_df.to_dict('records')
            profit, revenue, yields, agent_df, event_df, limits, gantt, missed, fin_metrics = run_simulation_with_structure(
                crops=crops_dict,
                soils=soils_dict,
                fields_data=original_fields,
                machines_data=machines,
                workers_data=workers,
                storages_data=storages,
                operations=operations,
                tech_cards=tech_cards,
                weather_daily=weather_daily,
                optimal_areas=None,
                days=days,
                initial_water_limit=water_limit,
                initial_budget=budget_limit,
                constant_costs=0.0,
                enable_adaptation=True,
                resource_prices=resource_prices,
                economic_params=economic_params,
                risks_df=st.session_state.risks_df
            )
            current_results[scen_name] = {
                'profit': profit,
                'revenue': revenue,
                'yields': yields,
                'agent_df': agent_df,
                'event_df': event_df,
                'limits': limits,
                'gantt': gantt,
                'missed_operations': missed,
                'financial_metrics': fin_metrics
            }
        st.session_state.current_results = current_results

    with st.spinner("Запуск имитации текущей структуры БЕЗ адаптации..."):
        current_results_no_adapt = {}
        for scen_name, weather_df in weather_by_name.items():
            weather_daily = weather_df.to_dict('records')
            profit, revenue, yields, agent_df, event_df, limits, gantt, missed, fin_metrics = run_simulation_with_structure(
                crops=crops_dict,
                soils=soils_dict,
                fields_data=original_fields,
                machines_data=machines,
                workers_data=workers,
                storages_data=storages,
                operations=operations,
                tech_cards=tech_cards,
                weather_daily=weather_daily,
                optimal_areas=None,
                days=days,
                initial_water_limit=water_limit,
                initial_budget=budget_limit,
                constant_costs=0.0,
                enable_adaptation=False,
                resource_prices=resource_prices,
                economic_params=economic_params,
                risks_df=st.session_state.risks_df
            )
            current_results_no_adapt[scen_name] = {
                'profit': profit,
                'revenue': revenue,
                'yields': yields,
                'agent_df': agent_df,
                'event_df': event_df,
                'limits': limits,
                'gantt': gantt,
                'missed_operations': missed,
                'financial_metrics': fin_metrics
            }
        st.session_state.current_results_no_adapt = current_results_no_adapt

    st.success("Все симуляции завершены!")
    st.rerun()

if mode == "📊 Модель" and 'robust' in st.session_state and 'current_results' in st.session_state:
    robust = st.session_state.robust
    current_results = st.session_state.current_results
    current_results_no_adapt = st.session_state.get('current_results_no_adapt', {})
    weather_by_name = st.session_state.get('weather_by_name', {})

    view_scenario = st.session_state.get('selected_view_scenario', "Общий робастный план")

    optimization_used = robust.get('lp_profit', 0) > 0 or st.session_state.get('optimize_flag_prev', False)

    if view_scenario == "Общий робастный план":
        opt_water = robust.get('adjusted_water_limit', water_limit)
        opt_budget = robust.get('adjusted_budget', budget_limit)
        opt_machines = robust.get('adjusted_machines', {})
        opt_workers = robust.get('adjusted_workers', {})
        common_limits = True
    else:
        scen_res = robust['scenario_results'][view_scenario]
        opt_water = scen_res['limits']['water_limit']
        opt_budget = scen_res['limits']['budget_limit']
        opt_machines = scen_res['limits'].get('peak_machines', scen_res['limits'].get('machines', {}))
        opt_workers = scen_res['limits'].get('peak_workers', scen_res['limits'].get('workers', {}))
        common_limits = False

    tabs = st.tabs(["📋 План (структура и ресурсы)", "🔄 Адаптации", "⚠️ Без адаптации", "💰 Финансы"])

    with tabs[0]:
        if 'phenology_warnings' in st.session_state and st.session_state.phenology_warnings:
            st.subheader("⚠️ Фенологические предупреждения")
            for scen, warns in st.session_state.phenology_warnings.items():
                with st.expander(f"Сценарий: {scen}"):
                    for crop, warn_msg in warns.items():
                        st.markdown(f"**{crop}:** <span style='color:red'>{warn_msg}</span>", unsafe_allow_html=True)
            st.write("Причины: неблагоприятная погода или пороги перехода между фазами GDD")
            st.markdown('---')
        st.subheader("📐 Оптимальная структура посевов")
        df_areas = pd.DataFrame({'Культура': list(robust['optimal_areas'].keys()), 'Площадь (га)': list(robust['optimal_areas'].values())}).round(2)
        st.dataframe(df_areas)
        if robust.get('lp_profit', 0) > 0:
            st.metric("Ожидаемая прибыль (LP)", f"{robust['lp_profit']:,.0f} руб")
        st.subheader("🛡️ Ресурсные лимиты (пиковая потребность)")
        st.write(f"**Вода:** {opt_water:,.0f} м³")
        st.write(f"**Бюджет:** {opt_budget:,.0f} руб")
        st.write("**Техника (пиковая потребность):**")
        for mt, cnt in opt_machines.items():
            st.write(f"  - {mt}: {cnt}")
        st.write("**Рабочие (пиковая потребность):**")
        for qual, cnt in opt_workers.items():
            st.write(f"  - {qual}: {cnt}")

        if optimization_used and view_scenario != "Общий робастный план":
            st.subheader("📊 Сравнение текущего и оптимального планов")
            comp_scen = view_scenario
            opt_res = robust['scenario_results'][comp_scen]
            curr_res = current_results.get(comp_scen)
            if curr_res is None:
                st.warning(f"Нет данных текущего плана для сценария {comp_scen}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"#### 📋 Исходный план – сценарий {comp_scen}")
                    st.write(f"**Прибыль:** {curr_res['profit']:,.0f} руб")
                    st.metric("Выручка (без затрат)", f"{curr_res['revenue']:,.0f} руб")
                    st.write(f"**Валовой сбор (т):** {sum(curr_res['yields'].values()):.1f}")
                    st.write("**Урожайность (т/га):**")
                    for crop, yld in curr_res['yields'].items():
                        st.write(f"  - {crop}: {yld:.2f}")
                with col2:
                    st.markdown(f"#### 📐 Оптимальный план – сценарий {comp_scen}")
                    st.write(f"**Прибыль:** {opt_res['profit']:,.0f} руб")
                    st.metric("Выручка (без затрат)", f"{opt_res['revenue']:,.0f} руб")
                    delta = opt_res['profit'] - curr_res['profit']
                    st.write(f"**Изменение:** {delta:+,.0f} руб")
                    st.write(f"**Валовой сбор (т):** {sum(opt_res['yields'].values()):.1f}")
                    st.write("**Урожайность (т/га):**")
                    for crop, yld in opt_res['yields'].items():
                        st.write(f"  - {crop}: {yld:.2f}")

                # График валового сбора
                gross_curr = {c: yld * robust['optimal_areas'].get(c,0) for c, yld in curr_res['yields'].items()}
                gross_opt = {c: yld * robust['optimal_areas'].get(c,0) for c, yld in opt_res['yields'].items()}
                df_gross = pd.DataFrame({
                    'Культура': list(set(gross_curr.keys()) | set(gross_opt.keys())),
                    'Текущий план (т)': [gross_curr.get(c,0) for c in list(set(gross_curr.keys()) | set(gross_opt.keys()))],
                    'Оптимальный план (т)': [gross_opt.get(c,0) for c in list(set(gross_curr.keys()) | set(gross_opt.keys()))]
                })
                fig_gross = px.bar(df_gross, x='Культура', y=['Текущий план (т)', 'Оптимальный план (т)'], barmode='group', title="Сравнение валового сбора")
                st.plotly_chart(fig_gross, key="gross_comparison")
        elif not optimization_used:
            st.info("Оптимизация не выполнялась. Для сравнения планов включите оптимизацию структуры.")
        else:
            st.info("Для сравнения выберите конкретный сценарий (не 'Общий робастный план').")

        st.subheader("🌦️ Устойчивость к погодным сценариям")
        profit_list = [res['profit'] for res in robust['scenario_results'].values()]
        if profit_list:
            st.write(f"**Средняя прибыль:** {np.mean(profit_list):,.0f} руб")
            st.write(f"**Стандартное отклонение:** {np.std(profit_list):,.0f} руб")
            st.write(f"**Минимум:** {min(profit_list):,.0f} руб, **Максимум:** {max(profit_list):,.0f} руб")

        st.subheader("🌾 Урожайность культур по сценариям")
        yields_by_scen = {}
        for scen, res in robust['scenario_results'].items():
            yields_by_scen[scen] = res['yields']
        fig_yield = plot_yield_by_scenario(yields_by_scen)
        if fig_yield and len(fig_yield.data) > 0:
            st.plotly_chart(fig_yield, key="yield_by_scenario")

        st.subheader("💰 Прибыль по сценариям")
        profit_df = pd.DataFrame([{'Сценарий': scen, 'Прибыль (руб)': res['profit']} for scen, res in robust['scenario_results'].items()])
        fig_profit = px.bar(profit_df, x='Сценарий', y='Прибыль (руб)', title="Прибыль по сценариям", color='Сценарий')
        st.plotly_chart(fig_profit, key="profit_by_scenario")

    
    with tabs[1]:
        if view_scenario != "Общий робастный план":
            st.subheader(f"🔄 Симуляция с адаптацией – сценарий «{view_scenario}»")
            res = robust['scenario_results'][view_scenario]
            event_df = res.get('event_df')

            if view_scenario in weather_by_name:
                fig_weather = plot_weather(weather_by_name[view_scenario], title=f"Погодные условия ({view_scenario})")
                if fig_weather:
                    st.plotly_chart(fig_weather, key=f"weather_adapt_{view_scenario}")

            agent_df = res.get('agent_df')
            if agent_df is not None and not agent_df.empty:
                if isinstance(agent_df.columns, pd.MultiIndex):
                    flat = agent_df.stack().reset_index()
                    flat.columns = ['Step', 'AgentID', 'Variable', 'Value']
                    flat = flat.pivot(index=['Step', 'AgentID'], columns='Variable', values='Value').reset_index()
                else:
                    flat = agent_df.reset_index()
                    if 'Step' not in flat.columns:
                        flat = flat.rename(columns={'index': 'Step'})
                    if 'AgentID' not in flat.columns:
                        if 'level_1' in flat.columns:
                            flat = flat.rename(columns={'level_1': 'AgentID'})
                        else:
                            flat['AgentID'] = flat.index

                fig_soil = plot_soil_moisture(agent_df)
                if fig_soil:
                    st.plotly_chart(fig_soil, key=f"soil_adapt_{view_scenario}")

                fig_mach = plot_machine_utilization(agent_df)
                if fig_mach:
                    st.plotly_chart(fig_mach, key=f"mach_adapt_{view_scenario}")

                # GDD и фазы
                if 'GDD' in flat.columns and 'Phase' in flat.columns:
                    valid = flat.dropna(subset=['GDD'])
                    if not valid.empty:
                        field_ids = valid['AgentID'].unique()
                        selected_field = st.selectbox("Выберите поле для фенологии (GDD)", field_ids, key="gdd_field_adapt")
                        crop_name = None
                        if 'CropType' in flat.columns:
                            crop_info = flat[flat['AgentID'] == selected_field]['CropType'].dropna()
                            if not crop_info.empty:
                                crop_name = crop_info.iloc[0]
                        crop_obj = st.session_state.get('crops_dict', {}).get(crop_name) if crop_name else None
                        fig_gdd = plot_gdd_and_phases(flat, selected_field, crop_obj)
                        if fig_gdd and len(fig_gdd.data) > 0:
                            st.plotly_chart(fig_gdd, key=f"gdd_adapt_{selected_field}")
                        
                        if view_scenario in weather_by_name:
                            weather_df_adapt = weather_by_name[view_scenario]

                            flat_agent = _flatten_agent_df(res['agent_df'])
                            if crop_obj and not flat_agent.empty:
                                fig_stress = plot_daily_stress_and_yield(
                                    flat_agent, weather_df_adapt, crop_obj, selected_field,
                                    field_capacity=350, wilting_point=150
                                )
                                if fig_stress and len(fig_stress.data) > 0:
                                    st.plotly_chart(fig_stress, key=f"stress_adapt_{selected_field}_{view_scenario}")
            
            gantt_actual = res.get('gantt', [])
            if gantt_actual:
                st.subheader("📊 Фактическая диаграмма Ганта (с адаптацией)")
                df_gantt_actual = pd.DataFrame(gantt_actual)
                fig_gantt_actual = plot_gantt_by_operations(df_gantt_actual)
                st.plotly_chart(fig_gantt_actual, use_container_width=True)

            st.subheader("📈 Адаптация лимитов воды и бюджета")
            if event_df is not None and not event_df.empty:
                # Вода
                water_events = event_df[event_df['event_type'] == 'Адаптация'].copy()
                water_events = water_events[water_events['description'].str.contains('вода', case=False, na=False)]
                if not water_events.empty:
                    changes = []
                    current = water_limit
                    changes.append((0, current))
                    for _, row in water_events.iterrows():
                        match = re.search(r'(\d+(?:\.\d+)?)\s*м³', row['description'])
                        if match:
                            delta = float(match.group(1))
                            current += delta
                            changes.append((row['day'], current))
                    days_plot = [c[0] for c in changes]
                    limits_plot = [c[1] for c in changes]
                    fig_water = go.Figure()
                    fig_water.add_trace(go.Scatter(x=days_plot, y=limits_plot, mode='lines+markers',
                                                   line=dict(shape='hv', color='blue'),
                                                   name='Лимит воды'))
                    fig_water.add_hline(y=water_limit, line_dash="dash", line_color="gray",
                                        annotation_text="Начальный лимит")
                    final_used = res['limits'].get('water_used', res['limits'].get('water_limit', current))
                    if final_used < current:
                        fig_water.add_hline(y=final_used, line_dash="dot", line_color="red",
                                            annotation_text=f"Фактически использовано: {final_used:.0f} м³")
                    fig_water.update_layout(title="Изменение лимита воды", xaxis_title="День", yaxis_title="Лимит, м³")
                    st.plotly_chart(fig_water, key="water_step_adapt")
                else:
                    st.info("Нет адаптаций воды")

                budget_events = event_df[event_df['event_type'] == 'Адаптация'].copy()
                budget_events = budget_events[budget_events['description'].str.contains('бюджет', case=False, na=False)]
                if not budget_events.empty:
                    changes = []
                    current = budget_limit
                    changes.append((0, current))
                    for _, row in budget_events.iterrows():
                        match = re.search(r'(\d+(?:\.\d+)?)\s*руб', row['description'])
                        if match:
                            delta = float(match.group(1))
                            current += delta
                            changes.append((row['day'], current))
                    days_plot = [c[0] for c in changes]
                    limits_plot = [c[1] for c in changes]
                    fig_budget = go.Figure()
                    fig_budget.add_trace(go.Scatter(x=days_plot, y=limits_plot, mode='lines+markers',
                                                    line=dict(shape='hv', color='green'),
                                                    name='Лимит бюджета'))
                    fig_budget.add_hline(y=budget_limit, line_dash="dash", line_color="gray",
                                         annotation_text="Начальный лимит")
                    final_used = res['limits'].get('budget_used', res['limits'].get('budget_limit', current))
                    if final_used < current:
                        fig_budget.add_hline(y=final_used, line_dash="dot", line_color="red",
                                             annotation_text=f"Фактически использовано: {final_used:.0f} руб")
                    fig_budget.update_layout(title="Изменение лимита бюджета", xaxis_title="День", yaxis_title="Лимит, руб")
                    st.plotly_chart(fig_budget, key="budget_step_adapt")
                else:
                    st.info("Нет адаптаций бюджета")

            st.subheader("📋 Таблица адаптаций")
            if event_df is not None and not event_df.empty:
                adapt_table = display_detailed_adaptation_table(event_df, st.session_state.get('operations_dict', {}), st.session_state.tech_cards_df)
                if not adapt_table.empty:
                    st.dataframe(adapt_table)
                else:
                    st.info("Нет деталей адаптаций")
            else:
                st.info("Нет событий адаптации")

            st.subheader("🏭 Пиковая потребность")
            peak_machines = res.get('limits', {}).get('peak_machines', {})
            peak_workers = res.get('limits', {}).get('peak_workers', {})
            if peak_machines:
                df_peak = pd.DataFrame([{'Тип техники': k, 'Максимальное количество': v} for k, v in peak_machines.items()])
                st.dataframe(df_peak)
            if peak_workers:
                df_peak_w = pd.DataFrame([{'Квалификация': k, 'Максимальное количество': v} for k, v in peak_workers.items()])
                st.dataframe(df_peak_w)
                
            water_usage = res['limits'].get('water_usage_by_field', [])
            if water_usage:
                st.subheader("💧 Водопотребление")
                df_water = pd.DataFrame(water_usage)
                st.dataframe(df_water, use_container_width=True)
            else:
                st.info("Нет данных о водопотреблении (возможно, полив не выполнялся).")
            
            st.subheader("📜 Журнал событий")
            if event_df is not None and not event_df.empty:
                event_types = event_df['event_type'].unique().tolist()
                selected_types = st.multiselect(
                    "Фильтровать типы событий",
                    options=event_types,
                    default=event_types,
                    key="event_filter"
                )
                filtered_df = event_df[event_df['event_type'].isin(selected_types)]
                st.dataframe(filtered_df[['day', 'event_type', 'agent_id', 'description']])
            else:
                st.info("Нет событий")

            st.subheader("⚠️ Риски")
            if 'event_df' in locals() and event_df is not None and not event_df.empty:
                risk_events = event_df[event_df['event_type'] == 'Риск применён']
                if not risk_events.empty:
                    st.dataframe(risk_events[['day', 'agent_id', 'description']])
                else:
                    st.info("Риски не активировались")
            else:
                st.info("Нет данных о событиях.")
        else:
            st.info("Для отображения адаптаций выберите конкретный сценарий в боковой панели.")


    with tabs[2]:
        if view_scenario != "Общий робастный план":
            st.header(f"⚠️ Результаты симуляции БЕЗ адаптации – сценарий «{view_scenario}»")
            st.subheader("⏰ Операции, выполненные с задержкой (окно нарушено)")
            no_adapt_res = current_results_no_adapt.get(view_scenario)
            if no_adapt_res is not None:
                gantt_no = no_adapt_res.get('gantt', [])
                if gantt_no:
                    df_fact = pd.DataFrame(gantt_no)
                    if 'planned_window_end' in df_fact.columns:
                        df_fact['shift_days'] = df_fact['end_day'] - df_fact['planned_window_end']
                        st.dataframe(df_fact[['operation_name', 'field_id', 'planned_window_end', 'start_day', 'end_day', 'shift_days']])
                    else:
                        st.info("В данных Gantt отсутствует planned_window_end")

                delayed_ops = no_adapt_res.get('limits', {}).get('delayed_operations', [])
                canceled_ops = no_adapt_res.get('limits', {}).get('canceled_operations', [])
                if canceled_ops:
                    df_canceled = pd.DataFrame(canceled_ops)
                    st.subheader("❌ Операции, отменённые из-за превышения допустимой задержки (>20 дней)")
                    st.dataframe(df_canceled)
                else:
                    st.success("✅ Нет отменённых операций")

                st.subheader("📜 Журнал событий (без адаптации)")
                event_df_no = no_adapt_res.get('event_df')
                if event_df_no is not None and not event_df_no.empty:
                    event_types = event_df_no['event_type'].unique().tolist()
                    selected_types = st.multiselect(
                        "Фильтровать типы событий",
                        options=event_types,
                        default=event_types,
                        key="event_filter_no_adapt"
                    )
                    filtered_df = event_df_no[event_df_no['event_type'].isin(selected_types)]
                    st.dataframe(filtered_df[['day', 'event_type', 'agent_id', 'description']])
                else:
                    st.info("Нет событий")

                st.subheader("📉 Причины отказов (без адаптации)")
                fig_rej = plot_rejection_reasons(event_df_no)
                if fig_rej and len(fig_rej.data) > 0:
                    st.plotly_chart(fig_rej, use_container_width=True)
                else:
                    st.success("Нет отказов – все операции выполнены")

                st.subheader("Простои из-за переувлажнения")
                fig_downtime = plot_downtime_events(event_df_no)
                if fig_downtime and len(fig_downtime.data) > 0:
                    st.plotly_chart(fig_downtime, use_container_width=True)
                else:
                    st.info("Нет простоев")
            else:
                st.info("Сначала запустите симуляцию (она выполняется автоматически).")
        else:
            st.info("Для отображения результатов без адаптации выберите конкретный сценарий в боковой панели.")

    with tabs[3]:
        st.header("Финансы")
        if view_scenario != "Общий робастный план":
            fin = robust['scenario_results'][view_scenario].get('financial_metrics', {})
            st.subheader(f"Сценарий: {view_scenario}")
        else:
            fin_list = [res.get('financial_metrics', {}) for res in robust['scenario_results'].values()]
            if fin_list:
                fin = {k: np.mean([f.get(k, 0) for f in fin_list]) for k in fin_list[0].keys()}
                st.subheader("Средние показатели по всем сценариям")
            else:
                fin = {}
        if fin:
            col1, col2, col3 = st.columns(3)
            storage = fin.get('storage_cost', 0)
            with col1:
                st.metric("Выручка", f"{fin.get('revenue', 0):,.0f} руб")
                st.metric("Переменные затраты", f"{fin.get('total_variable_cost', 0):,.0f} руб")
                if storage>0:
                    st.metric("Затраты на хранение", f"{storage:,.0f} руб") 
            with col2:
                st.metric("Административные расходы", f"{fin.get('admin_cost', 0):,.0f} руб")
                st.metric("Доп. затраты от рисков", f"{fin.get('extra_risk_cost', 0):,.0f} руб")
                st.metric("EBITDA", f"{fin.get('ebitda', 0):,.0f} руб")
                st.metric("Амортизация", f"{fin.get('depreciation', 0):,.0f} руб")
            with col3:
                st.metric("Налог на имущество", f"{fin.get('property_tax', 0):,.0f} руб")
                st.metric("Прибыль до налога", f"{fin.get('profit_before_tax', 0):,.0f} руб")
                st.metric("Чистая прибыль", f"{fin.get('net_profit', 0):,.0f} руб", delta_color="normal")

            base_var = fin.get('total_variable_cost', 0) - fin.get('extra_risk_cost', 0)
            risk_cost = fin.get('extra_risk_cost', 0)
            storage = fin.get('storage_cost', 0)
            admin = fin.get('admin_cost', 0)
            cost_breakdown = {}
            if base_var > 0:
                cost_breakdown["Основные переменные затраты"] = base_var
            if risk_cost > 0:
                cost_breakdown["Затраты от рисков"] = risk_cost
            if storage > 0:
                cost_breakdown["Хранение"] = storage
            if admin > 0:
                cost_breakdown["Административные"] = admin

            if cost_breakdown:
                fig_pie = px.pie(names=list(cost_breakdown.keys()), values=list(cost_breakdown.values()),
                         title="Структура затрат (без амортизации)")
                st.plotly_chart(fig_pie, key="cost_structure")
            else:
                st.info("Нет данных для построения диаграммы затрат")
            

            st.subheader("Постоянные затраты и налоги")
            st.write(f"**Амортизация:** {fin.get('depreciation',0):,.0f} руб")
            st.write(f"**Налог на имущество:** {fin.get('property_tax',0):,.0f} руб")
            st.write(f"**Налог на прибыль:** {fin.get('income_tax',0):,.0f} руб")
        else:
            st.info("Финансовые метрики не доступны. Убедитесь, что симуляция завершена и economic_params загружены.")
else:
    st.header("📋 Редактирование данных")
    tab_names = [
        "Культуры", "Фазы", "Почвы", "Поля", "Техника",
        "Рабочие", "Хранилища", "Операции", "Техкарты", "Погода", "Сценарии", "Ограничения","Цены", "Экономика", "Риски"
        ]
    config_keys = ["crops", "phases", "soils", "fields", "machines",
                   "workers", "storages", "operations", "tech_cards",
                   "weather", "scenarios", "constraints", "prices","economic_params", "risks"]

    data_tabs = st.tabs(tab_names)
    for tab, key in zip(data_tabs, config_keys):
        render_data_tab(tab, data_configs[key])