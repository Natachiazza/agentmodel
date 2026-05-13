import mesa
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pulp as pl
import math
import plotly.graph_objects as go
import plotly.express as px


class CropPhase(Enum):
    FALLOW = "Пар"
    SOWING = "SOWING"
    PLANTED = "PLANTED"
    BUD_BREAK = "BUD_BREAK"
    FLOWERING = "FLOWERING"
    FRUIT_SET = "FRUIT_SET"
    VERAISON = "VERAISON"
    BERRY_GROWTH = "BERRY_GROWTH"
    RIPENING = "RIPENING"
    HARVESTED = "HARVESTED"
    GERMINATION = "GERMINATION"
    TILLERING = "TILLERING"
    STEM_ELONGATION = "STEM_ELONGATION"
    HEADING = "HEADING"
    GRAIN_FILL = "GRAIN_FILL"
    MATURE = "MATURE"
    VEGETATIVE = "VEGETATIVE"
    BUDDING = "BUDDING"
    SEED_MATURITY = "SEED_MATURITY"
    SENESCENCE = "SENESCENCE"
    DORMANCY = "DORMANCY"

class MachineType(Enum):
    TRACTOR = "tractor"
    HARVESTER = "harvester"
    IRRIGATOR = "irrigator"
    SPRAYER = "sprayer"
    TRANSPORTER = "transporter"

class MachineStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    MOVING = "moving"
    BROKEN = "broken"

class TaskType(Enum):
    PLOW = "plow"
    CULTIVATE = "cultivate"
    SOW = "sow"
    IRRIGATE = "irrigate"
    SPRAY = "spray"
    HARVEST = "harvest"
    TRANSPORT = "transport"

class CropPhaseInfo:
    def __init__(self, phase_enum: str, gdd_required: float, kc: float, sensitivity: float,
                 next_phase: str = None, weights: Dict[str, float] = None,
                 Tmin: float = None, Topt1: float = None, Topt2: float = None, Tmax: float = None,
                 P_min: float = None, P_opt1: float = None, P_opt2: float = None, P_max: float = None,
                 PAR_min: float = None, PAR_opt1: float = None, PAR_opt2: float = None, PAR_max: float = None):
        self.phase_enum = phase_enum
        self.gdd_required = gdd_required
        self.kc = kc
        self.sensitivity = sensitivity
        self.next_phase = next_phase
        self.weights = weights or {"temp": 0.5, "water": 0.3, "par": 0.2}
        self.Tmin = Tmin
        self.Topt1 = Topt1
        self.Topt2 = Topt2
        self.Tmax = Tmax
        self.P_min = P_min
        self.P_opt1 = P_opt1
        self.P_opt2 = P_opt2
        self.P_max = P_max
        self.PAR_min = PAR_min
        self.PAR_opt1 = PAR_opt1
        self.PAR_opt2 = PAR_opt2
        self.PAR_max = PAR_max

class Crop:
    def __init__(self, row: dict, phases: Dict[str, CropPhaseInfo]):
        self.crop_id = row['crop_id']
        self.name = row['name']
        self.type = row.get('type', 'field')
        self.T_base = row['T_base']
        self.T_min = row['T_min']
        self.T_opt1 = row['T_opt1']
        self.T_opt2 = row['T_opt2']
        self.T_max = row['T_max']
        self.yield_pot = row['yield_pot']
        self.price = row['price']
        self.seed_cost = row['seed_cost']
        self.water_need = row['water_need']
        self.irrigation_threshold = row['irrigation_threshold']
        self.irrigation_amount = row['irrigation_amount']
        self.max_share = row['max_share']
        self.sowing_start = row['sowing_start']
        self.sowing_end = row['sowing_end']
        self.harvest_start = row['harvest_start']
        self.harvest_end = row['harvest_end']
        self.phases = phases
        self.establishment_cost = row.get('establishment_cost', 0.0)
        self.useful_life_years = row.get('useful_life_years', 20)

class Soil:
    def __init__(self, row: dict):
        self.soil_id = row['soil_id']
        self.name = row['name']
        self.field_capacity = row['field_capacity']
        self.wilting_point = row['wilting_point']
        self.infiltration_rate = row['infiltration_rate']
        self.bulk_density = row['bulk_density']
        self.organic_matter = row['organic_matter']

class Operation:
    def __init__(self, row: dict):
        self.operation_id = row['operation_id']
        self.name = row['name']
        self.task_type = row['task_type'] if isinstance(row['task_type'], TaskType) else TaskType(row['task_type'])
        self.machine_type = row['machine_type'] if isinstance(row['machine_type'], MachineType) else MachineType(row['machine_type'])
        self.duration_per_ha = row['duration_per_ha']
        self.fuel_per_ha = row['fuel_per_ha']
        self.material_consumption = row.get('material_consumption', 0)
        self.material_type = row.get('material_type')
        self.max_soil_moisture = row.get('max_soil_moisture', 9999)
        self.requires_operator = row.get('requires_operator', True)

class TechCard:
    def __init__(self, row: dict):
        self.tech_card_id = row.get('tech_card_id', 0)
        self.crop_id = row['crop_id']
        self.phase_enum = row.get('phase_enum')
        self.operation_id = row['operation_id']
        self.day_start = row['day_start']
        self.day_end = row['day_end']
        self.priority = row['priority']
        self.resource_id = row.get('resource_id')
        self.amount_per_ha = row.get('amount_per_ha', 0)

def temperature_stress(temp: float, crop: Crop = None, phase_info: CropPhaseInfo = None) -> float:
    if phase_info and phase_info.Tmin is not None:
        Tmin, Topt1, Topt2, Tmax = phase_info.Tmin, phase_info.Topt1, phase_info.Topt2, phase_info.Tmax
    elif crop:
        Tmin, Topt1, Topt2, Tmax = crop.T_min, crop.T_opt1, crop.T_opt2, crop.T_max
    else:
        return 1.0
    if temp < Tmin or temp > Tmax:
        return 0.0
    if temp < Topt1:
        return (temp - Tmin) / (Topt1 - Tmin)
    if temp <= Topt2:
        return 1.0
    return (Tmax - temp) / (Tmax - Topt2)

def water_stress(sm: float, fc: float, wp: float) -> float:
    if sm >= fc:
        return 1.0
    if sm <= wp:
        return 0.0
    return (sm - wp) / (fc - wp)

def precip_stress(precip: float, phase_info: CropPhaseInfo = None, crop: Crop = None) -> float:
    if phase_info and phase_info.P_min is not None:
        Pmin, Popt1, Popt2, Pmax = phase_info.P_min, phase_info.P_opt1, phase_info.P_opt2, phase_info.P_max
    else:
        return 1.0
    if precip < Pmin or precip > Pmax:
        return 0.0
    if precip < Popt1:
        return (precip - Pmin) / (Popt1 - Pmin)
    if precip <= Popt2:
        return 1.0
    return (Pmax - precip) / (Pmax - Popt2)

def par_stress_with_phase(par: float, phase_info: CropPhaseInfo = None, crop: Crop = None) -> float:
    if phase_info and phase_info.PAR_min is not None:
        PARmin, PARopt1, PARopt2, PARmax = phase_info.PAR_min, phase_info.PAR_opt1, phase_info.PAR_opt2, phase_info.PAR_max
    else:
        return 1.0
    if par < PARmin or par > PARmax:
        return 0.0
    if par < PARopt1:
        return (par - PARmin) / (PARopt1 - PARmin)
    if par <= PARopt2:
        return 1.0
    return (PARmax - par) / (PARmax - PARopt2)

def calc_gdd(tmin: float, tmax: float, base_temp: float) -> float:
    avg = (tmin + tmax) / 2
    return max(0.0, avg - base_temp)

class FieldAgent(Agent):
    def __init__(self, unique_id: int, model, field_data: dict, soil: Soil, crop: Crop = None):
        super().__init__(unique_id, model)
        self.area = field_data['area']
        self.soil = soil
        self.crop_type = field_data.get('crop_plan')
        self.crop = crop
        self.phase = CropPhase.FALLOW if crop is None else CropPhase.SOWING
        self.phase_start_day = model.day_of_year
        self.gdd_accumulated = 0.0
        self.soil_moisture = field_data.get('initial_moisture', self.soil.field_capacity)
        self.nutrients = {
            'N': field_data.get('initial_N', 50),
            'P': field_data.get('initial_P', 30),
            'K': field_data.get('initial_K', 40)
        }
        self.harvest_ready = False
        self.pos_x = field_data['x']
        self.pos_y = field_data['y']
        self.current_phase_stress_sum = 0.0
        self.current_phase_days = 0
        self.phase_productivity_factor = 1.0
        self.active_machines = 0
        self.completed_operations = set()
        self.missed_operations = []   # список operation_id  не выполненных в окне
        self.delayed_operations = []   # (op_id, days_delayed)
        self.actual_yield = 0.0
        self.irrigation_today = 0.0
        self.total_water_used = 0.0
        self.total_fuel_used = 0.0
        self.total_material_used = 0.0
        self.total_labor_cost = 0.0 
        self.is_busy = False
        self.current_request = None
        self.consecutive_cold_days = 0 
        self.consecutive_rain_days = 0
        self.yield_loss_factor = 1.0 

        if self.crop and self.crop.phases:
            first_phase_enum = min(self.crop.phases.keys(), key=lambda k: self.crop.phases[k].gdd_required)
            self.first_phase_enum = first_phase_enum
            self.first_phase_gdd = self.crop.phases[first_phase_enum].gdd_required
        else:
            self.first_phase_enum = None
            self.first_phase_gdd = None

    def step(self):
        if self.crop is None:
            return
        
        weather = self.model.weather
        day = self.model.day_of_year
        tmin, tmax = weather.tmin, weather.tmax
        tmean = (tmin + tmax) / 2
        self.check_risks(weather)

        # Водный баланс
        Ra = 15 + 25 * np.sin(2 * np.pi * (day - 80) / 365)
        Ra = max(10, Ra)
        et0 = 0.0023 * (tmean + 17.8) * (tmax - tmin) ** 0.5 * Ra
        phase_info = self.crop.phases.get(self.phase.value) if self.phase != CropPhase.SOWING else None
        kc = phase_info.kc if phase_info else 0.5
        ks = water_stress(self.soil_moisture, self.soil.field_capacity, self.soil.wilting_point)
        et = et0 * kc * ks
        rain = weather.precipitation
        irrigation = self.irrigation_today
        self.irrigation_today = 0.0
        sm_temp = self.soil_moisture + rain + irrigation - et
        if sm_temp > self.soil.field_capacity:
            drainage = sm_temp - self.soil.field_capacity
            self.soil_moisture = self.soil.field_capacity
        else:
            self.soil_moisture = max(self.soil.wilting_point, sm_temp)

        # Фенология
        gdd = calc_gdd(tmin, tmax, self.crop.T_base)
        self.gdd_accumulated += gdd

        if self.phase == CropPhase.SOWING and self.first_phase_gdd and self.gdd_accumulated >= self.first_phase_gdd:
            old_phase = self.phase.value
            try:
                new_phase = CropPhase[self.first_phase_enum]
                self.phase = new_phase
                self.phase_start_day = day
                self.gdd_accumulated = 0.0
                self.current_phase_stress_sum = 0.0
                self.current_phase_days = 0
                self.model.log_event(self.unique_id, "Смена фазы", f"из {old_phase} в {new_phase.value}, GDD={gdd:.1f}")
            except KeyError:
                pass

        if self.phase != CropPhase.SOWING:
            phase_info = self.crop.phases.get(self.phase.value)
            if phase_info and phase_info.gdd_required > 0 and self.gdd_accumulated >= phase_info.gdd_required:
                if phase_info.next_phase:
                    try:
                        new_phase = CropPhase[phase_info.next_phase]
                        if self.current_phase_days > 0:
                            avg_stress = self.current_phase_stress_sum / self.current_phase_days
                            self.phase_productivity_factor *= avg_stress
                        self.current_phase_stress_sum = 0.0
                        self.current_phase_days = 0
                        old_phase = self.phase.value
                        self.phase = new_phase
                        self.phase_start_day = day
                        self.model.log_event(self.unique_id, "Смена фазы", f"из {old_phase} в {new_phase.value}, GDD={self.gdd_accumulated:.1f}")
                    except KeyError:
                        pass

        # Стресс
        if self.phase not in (CropPhase.SOWING, CropPhase.HARVESTED, CropPhase.FALLOW):
            phase_info = self.crop.phases.get(self.phase.value)
            stress_temp = temperature_stress(weather.temperature, crop=self.crop, phase_info=phase_info)
            stress_water = ks
            stress_par = par_stress_with_phase(weather.par, phase_info=phase_info, crop=self.crop)
            stress_precip = precip_stress(weather.precipitation, phase_info=phase_info, crop=self.crop)
            if phase_info and phase_info.weights:
                w = phase_info.weights
                total_weight = sum(w.values())
                if total_weight > 0:
                    daily_stress = (w.get('temp', 0) * stress_temp +
                                    w.get('water', 0) * stress_water +
                                    w.get('par', 0) * stress_par) / total_weight
                else:
                    daily_stress = 1.0
            else:
                daily_stress = (stress_temp + stress_water + stress_par) / 3.0
            self.current_phase_stress_sum += daily_stress
            self.current_phase_days += 1

        # Заявки на операции
        if self.phase not in (CropPhase.SOWING, CropPhase.FALLOW):
            ops = self.model.get_operations_for_crop_phase(self.crop.name, self.phase.value)
            for tc in ops:
                days_in_phase = day - self.phase_start_day
                if tc.day_start <= days_in_phase <= tc.day_end:
                    if (tc.operation_id, self.phase.value) not in self.completed_operations:
                        op = self.model.get_operation_by_id(tc.operation_id)
                        if op:
                            window_end_abs = day + (tc.day_end - days_in_phase)
                            self.model.add_request(field_id=self.unique_id, operation=op,
                                                   priority=tc.priority, window_end=window_end_abs,
                                                   total_area=self.area)
                            self.completed_operations.add((tc.operation_id, self.phase.value))

            # Полив
            irrig_ops = [tc for tc in ops if self.model.get_operation_by_id(tc.operation_id).task_type == TaskType.IRRIGATE]
            if irrig_ops and self.soil_moisture < self.crop.irrigation_threshold:
                sens = phase_info.sensitivity if phase_info else 0.5
                base_prio = 50
                alpha = 10
                if self.soil_moisture <= self.soil.wilting_point:
                    dynamic = 1.0
                else:
                    dynamic = (self.crop.irrigation_threshold - self.soil_moisture) / \
                              (self.crop.irrigation_threshold - self.soil.wilting_point) 
                prio = base_prio + alpha * dynamic * sens
                irrig_op = self.model.get_operation_by_type(TaskType.IRRIGATE)
                if irrig_op:
                    irrig_tc = irrig_ops[0]
                    days_in_phase = day - self.phase_start_day
                    window_end_abs = day + (irrig_tc.day_end - days_in_phase)
                    self.model.log_event(self.unique_id, "Потребность в поливе",
                                         f"влажность={self.soil_moisture:.1f}, порог={self.crop.irrigation_threshold}, приоритет={prio}")
                    self.model.add_request(self.unique_id, irrig_op, priority=int(prio),
                                           window_end=window_end_abs, total_area=self.area)

            # Уборка
            if not self.harvest_ready:
                if day < self.crop.harvest_start:
                    return
                ops = self.model.get_operations_for_crop_phase(self.crop.name, self.phase.value)
                harvest_tc = None
                for tc in ops:
                    op = self.model.get_operation_by_id(tc.operation_id)
                    if op and op.task_type == TaskType.HARVEST:
                        harvest_tc = tc
                        break
                if harvest_tc:
                    self.harvest_ready = True
                    harvest_op = self.model.get_operation_by_id(harvest_tc.operation_id)
                    window_end_abs = self.crop.harvest_end
                    self.model.add_request(self.unique_id, harvest_op, priority=100,
                                           window_end=window_end_abs, total_area=self.area)
        
        if day > self.crop.harvest_end and self.phase not in (CropPhase.HARVESTED, CropPhase.FALLOW):
            self.model.cleanup_field_requests(self.unique_id)
            base_yield = self.crop.yield_pot * self.phase_productivity_factor
            if self.harvest_ready and self.phase != CropPhase.HARVESTED:
                harvest_op = self.model.get_operation_by_type(TaskType.HARVEST)
                if harvest_op and harvest_op.operation_id not in self.missed_operations:
                    self.missed_operations.append(harvest_op.operation_id)
            
            self.actual_yield = self._compute_final_yield(base_yield)
            self.phase = CropPhase.HARVESTED
            self.harvest_ready = False
            self.is_busy = False 
            self.current_request = None
            self.model.log_event(self.unique_id, "Принудительная уборка", f"Урожай {self.actual_yield:.2f} т/га")

    def apply_operation(self, op: Operation, amount: float = None, fuel_cost: float = 0, material_cost: float = 0, labor_cost: float = 0, water_used: float = 0):
        if op.task_type == TaskType.IRRIGATE:
            self.irrigation_today = self.crop.irrigation_amount
            self.total_water_used += self.crop.irrigation_amount * self.area * 10.0
            self.model.log_event(self.unique_id, "Полив", f"Добавлено {self.crop.irrigation_amount} мм")
        elif op.task_type == TaskType.SPRAY:
            material_kg = (amount or op.material_consumption) * self.area
            self.total_material_used += material_cost
            self.model.log_event(self.unique_id, f"Расход {op.material_type}", 
                             f"{material_kg:.1f} кг (операция {op.name})")
            if op.material_type == 'N':
                self.nutrients['N'] += amount or op.material_consumption
        
        fuel_price = self.model.market.get_price('fuel')
        fuel_liters = fuel_cost / fuel_price if fuel_price > 0 else 0
        self.total_fuel_used += fuel_liters
        self.model.log_event(self.unique_id, "Расход топлива", 
                         f"{fuel_liters:.1f} л (операция {op.name})")
        
        self.total_material_used += material_cost
        self.total_labor_cost += labor_cost
        self.model.log_event(self.unique_id, "Трудозатраты", 
                         f"{labor_cost:.1f} чел·ч (операция {op.name})")

    def _compute_final_yield(self, base_yield: float) -> float:
        penalty_missed = 1.0
        for op_id in self.missed_operations:
            op = self.model.get_operation_by_id(op_id)
            if op.task_type == TaskType.SPRAY:
                penalty_missed *= 0.3
            elif op.task_type == TaskType.CULTIVATE:
                penalty_missed *= 0.2
            elif op.task_type == TaskType.PLOW:
                penalty_missed *= 0.3
            elif op.task_type == TaskType.SOW:
                penalty_missed = 0.3
                break
            elif op.task_type == TaskType.IRRIGATE:
                penalty_missed *= 0.3 
            elif op.task_type == TaskType.HARVEST:
                penalty_missed = 0.0  
                break
        penalty_delay = 1.0
        weeks=0.0
        for op_id, delay in self.delayed_operations:
            weeks = delay / 7.0
            penalty_delay *= max(0.1, 1.0 - weeks * 0.15)
            
        day = self.model.day_of_year
        if day > self.crop.harvest_end:
            days_late = day - self.crop.harvest_end
            penalty_late = max(0.0, 1.0 - days_late / 14.0)
        else:
            penalty_late = 1.0

        base_y = base_yield * self.yield_loss_factor
        return base_y * penalty_missed * penalty_delay * penalty_late
    
    def finalize_harvest(self):
        if self.phase == CropPhase.HARVESTED:
            return
        if self.phase in (CropPhase.MATURE, CropPhase.HARVESTED):
            self.model.cleanup_field_requests(self.unique_id)
            base_yield = self.crop.yield_pot * self.phase_productivity_factor
            self.actual_yield = self._compute_final_yield(base_yield)
            self.phase = CropPhase.HARVESTED
            self.harvest_ready = False
            storage = self.model.get_storage_for('grain')
            if storage:
                storage.receive(self.actual_yield * self.area)
            self.model.log_event(self.unique_id, "Уборка завершена",
                                f"Урожайность {self.actual_yield:.2f} т/га, всего {self.actual_yield * self.area:.1f} т")
    
    def get_yield(self) -> float:
        return self.actual_yield * self.area if self.phase == CropPhase.HARVESTED else 0.0

    def check_risks(self, weather):
        for risk in self.model.risks:
            if risk.crop_type != 'all' and risk.crop_type != self.crop.type:
                continue
            if risk.phenophase != 'any' and risk.phenophase != self.phase.value:
                continue

            triggered = False
         #-3,2 -порог -3 C, нужно 2 дня подряд 
            if risk.trigger_type == 'temp_below_days':
                parts = list(map(float, risk.trigger_value.split(',')))
                threshold, required_days = parts[0], int(parts[1])
                if weather.tmin < threshold:
                    self.consecutive_cold_days += 1
                    if self.consecutive_cold_days >= required_days:
                        triggered = True
                else:
                        self.consecutive_cold_days = 0
            elif risk.trigger_type == 'rain_after_temp_days':
                parts = list(map(float, risk.trigger_value.split(',')))
                min_temp, min_rain, required_days = parts[0], parts[1], int(parts[2])
                cond = (weather.tmin > min_temp) and (weather.precipitation > min_rain)
                if cond:
                    self.consecutive_rain_days += 1
                    if self.consecutive_rain_days >= required_days:
                        triggered = True
                else:
                    self.consecutive_rain_days = 0

            elif risk.trigger_type == 'random':
                if np.random.random() < risk.probability:
                    triggered = True

            if triggered:
                self.apply_risk_effects(risk)

    def apply_risk_effects(self, risk):
        self.yield_loss_factor *= (1.0 - risk.penalty_yield)
        if risk.fixed_cost_rub > 0:
            self.model.total_variable_cost_inc += risk.fixed_cost_rub
   
        if risk.downtime_days > 0 and risk.trigger_type == 'random':
            for mt, machines in self.model.machines.items():
                for m in machines:
                    if m.current_field == self.unique_id and m.status == MachineStatus.BUSY:
                        m.break_down(risk.downtime_days)
                        break  
        self.model.log_event(self.unique_id, "Риск применён",
                         f"{risk.name}: потери урожая {risk.penalty_yield*100:.0f}%, доп. затраты {risk.fixed_cost_rub:.0f} руб")
            
                
class WeatherAgent(Agent):
    def __init__(self, unique_id: int, model, daily_data: List[dict] = None):
        super().__init__(unique_id, model)
        self.daily_data = daily_data
        self.tmin = 0.0
        self.tmax = 0.0
        self.temperature = 0.0
        self.precipitation = 0.0
        self.par = 0.0

    def step(self):
        day = self.model.day_of_year
        if self.daily_data and day <= len(self.daily_data):
            data = self.daily_data[day-1]
            self.tmin = data['tmin']
            self.tmax = data['tmax']
            self.temperature = (self.tmin + self.tmax) / 2
            self.precipitation = data['precip']
            self.par = data['par']
        else:
            self.temperature = 15 + 15 * np.sin(2 * np.pi * (day - 105) / 365)
            self.tmin = self.temperature - 5
            self.tmax = self.temperature + 5
            self.precipitation = np.random.exponential(5) if np.random.random() > 0.7 else 0
            self.par = 10 + 5 * np.sin(2 * np.pi * (day - 105) / 365)

class MachineAgent(Agent):
    def __init__(self, unique_id: int, model, machine_data: dict):
        super().__init__(unique_id, model)
        self.machine_type = machine_data['type'] if isinstance(machine_data['type'], MachineType) else MachineType(machine_data['type'])
        self.brand = machine_data['brand']
        self.productivity = machine_data['productivity']
        self.fuel_consumption = machine_data['fuel_consumption']
        self.fuel_capacity = machine_data['fuel_capacity']
        self.cost_hour = machine_data['cost_hour']
        self.requires_operator = machine_data.get('requires_operator', True)
        self.status = MachineStatus.IDLE
        self.current_field = None
        self.days_remaining = 0
        self.fuel = self.fuel_capacity
        self.days_to_repair = 0
        #для амортизации
        self.initial_cost = machine_data.get('initial_cost', 0.0)
        self.useful_life_years = machine_data.get('useful_life_years', 10)

    def step(self):
        if self.status == MachineStatus.BROKEN:
            self.days_to_repair -= 1
            if self.days_to_repair <= 0:
                self.status = MachineStatus.IDLE
                self.model.log_event(-1, "Ремонт завершён", f"Машина {self.unique_id} готова")
            return
        
        if self.status == MachineStatus.BUSY:
            self.days_remaining -= 1
            if self.days_remaining <= 0:
                if self.current_field is not None:
                    self.model.busy_fields.discard(self.current_field)
                    field = self.model.get_field(self.current_field)
                    if field:
                        field.active_machines -= 1
                    self.model.notify_operation_complete(self.current_field, self)
                    self.current_field = None
                self.status = MachineStatus.IDLE
                if self.requires_operator:
                    self.model.release_worker(self.unique_id)

    def assign_task(self, field_id: int, operation: Operation, area: float, days_needed: int):
        self.current_field = field_id
        self.days_remaining = 1 #!!!
        self.status = MachineStatus.BUSY

    def break_down(self, duration_days: int):
        if self.status == MachineStatus.BUSY:
        # прерываем текущую операцию
            if self.current_field:
                field = self.model.get_field(self.current_field)
                if field:
                    field.active_machines -= 1
                self.current_field = None
        self.status = MachineStatus.BROKEN
        self.days_to_repair = duration_days
        self.model.log_event(self.unique_id, "Поломка", f"Машина вышла из строя на {duration_days} дней")

class WorkerAgent(Agent):
    def __init__(self, unique_id: int, model, worker_data: dict):
        super().__init__(unique_id, model)
        self.name = worker_data['name']
        self.qualification = worker_data['qualification']
        self.cost_per_hour = worker_data['cost_per_hour']
        self.max_hours_per_day = worker_data['max_hours_per_day']
        self.status = "idle"
        self.assigned_machine = None
       
        self.hours_worked_today = 0
        self.last_day = 0

    def step(self):
        if self.model.day_of_year != self.last_day:
            self.hours_worked_today = 0
            self.last_day = self.model.day_of_year
            self.status = "idle"       # освобождаем
            self.assigned_machine = None

    def assign(self, hours: float, machine_id: int) -> bool:
        if self.hours_worked_today + hours > self.max_hours_per_day + 1e-6:
            return False
        self.hours_worked_today += hours
        self.status = "busy"
        self.assigned_machine = machine_id
        return True

    def release(self):
        self.status = "idle"
        self.assigned_machine = None

class MarketAgent(Agent):
    def __init__(self, unique_id, model, prices: dict):
        super().__init__(unique_id, model)
        self.prices = prices   # {'fuel': 60, 'water': 10, ...}

    def get_price(self, resource_name: str) -> float:
        return self.prices.get(resource_name, 0.0)
    
class Risk:
    def __init__(self, row: dict):
        self.risk_id = row['risk_id']
        self.name = row['name']
        self.crop_type = row['crop_type']
        self.phenophase = row['phenophase']
        self.trigger_type = row['trigger_type']
        self.trigger_value = row['trigger_value']
        self.penalty_yield = row['penalty_yield']
        self.fixed_cost_rub = row['fixed_cost_rub'] 
        self.downtime_days = row['downtime_days']    # дни простоя техники
        self.probability = row['probability']

class StorageAgent(Agent):
    def __init__(self, unique_id: int, model, storage_data: dict):
        super().__init__(unique_id, model)
        self.storage_type = storage_data['type']
        self.capacity = storage_data['capacity']
        self.current_stock = 0.0
        self.pos_x = storage_data['x']
        self.pos_y = storage_data['y']
        #параметры хранения
        self.cost_per_ton_per_day = storage_data.get('cost_per_ton_per_day', 0.0)
        self.annual_loss_rate = storage_data.get('annual_loss_rate', 0.0)
        # партии: {"amount": float, "days": int}
        self.batches = []

    def receive(self, amount: float):
        free = self.capacity - self.current_stock
        accepted = min(amount, free)
        if accepted < amount:
            self.model.log_event(self.unique_id, "Переполнение склада",
                                 f"Не принято {amount - accepted:.1f} т (лимит {self.capacity} т)")
        # добавляем партию
        if accepted > 0:
            self.batches.append({"amount": accepted, "days": 0})
            self.current_stock += accepted

    def withdraw(self, amount: float) -> float:
        can_take = min(amount, self.current_stock)
        self.current_stock -= can_take
        remaining = can_take
        new_batches = []
        for b in self.batches:
            if remaining <= 0:
                new_batches.append(b)
                continue
            take = min(b["amount"], remaining)
            b["amount"] -= take
            remaining -= take
            if b["amount"] > 0:
                new_batches.append(b)
        self.batches = new_batches
        return can_take
    
    def step(self):
        storage_cost_inc = self.current_stock * self.cost_per_ton_per_day
        self.model.total_storage_cost += storage_cost_inc
        #потери при хранении
        new_batches = []
        for b in self.batches:
            b["days"] += 1
            if b["days"] > 30:
                loss = b["amount"] * self.annual_loss_rate / 365.0
                if loss > 0:
                    b["amount"] -= loss
                    self.current_stock -= loss
                    self.model.log_event(self.unique_id, "Потери при хранении",
                                         f"Потеряно {loss:.2f} т, остаток {b['amount']:.2f} т")
            if b["amount"] > 0:
                new_batches.append(b)
        self.batches = new_batches

    def release(self, amount: float) -> float:
        released = 0.0
        while amount > 0 and self.batches:
            b = self.batches[0]
            take = min(b["amount"], amount)
            b["amount"] -= take
            released += take
            amount -= take
            self.current_stock -= take
            if b["amount"] <= 0:
                self.batches.pop(0)
        return released

class Request:
    def __init__(self, field_id: int, operation: Operation, priority: int, window_end: int, total_area: float):
        self.field_id = field_id
        self.operation = operation
        self.priority = priority
        self.window_end = window_end
        self.remaining_area = total_area
        self.start_day = None
        self.end_day = None
        self.assigned_machines = []
        self.workers_assigned = []
        self.daily_work = 0.0
        self.operation_started = False
        self.total_fuel_cost = 0.0
        self.total_material_cost = 0.0
        self.total_labor_cost = 0.0
        self.total_water_used = 0.0
        self.workers_needed = 0
        self.days_needed = 0
        self.logged_completion = False 
        self.window_broken = False   
        self.days_delayed = 0
        self.scheduled_today = False

class DispatcherAgent(Agent):
    def __init__(self, unique_id: int, model, enable_adaptation: bool = True):
        super().__init__(unique_id, model)
        self.enable_adaptation = enable_adaptation
        self.active_requests = []

    def add_request(self, field_id: int, operation: Operation, priority: int, window_end: int, total_area: float):
        if self.has_active_request(field_id, operation.operation_id):
            return
        req = Request(field_id, operation, priority, window_end, total_area)
        self.active_requests.append(req)

    def has_active_request(self, field_id: int, operation_id: int) -> bool:
        for req in self.active_requests:
            if req.field_id == field_id and req.operation.operation_id == operation_id:
                return True
        return False

    def step(self):
        model = self.model
        day = model.day_of_year
        for req in self.active_requests:
            req.scheduled_today = False

        # Удаление завершённых заявок
        for req in self.active_requests[:]:
            if req.scheduled_today:
                continue
            if req.remaining_area <= 0:
                for w in req.workers_assigned:
                    w.release()
                    model.log_event(req.field_id, "Освобождение рабочего",
                        f"Рабочий {w.name} ({w.qualification}) освобождён")
                
                req.end_day = day
                field = model.get_field(req.field_id)
                if field:
                    field.is_busy = False
                    field.current_request = None
                    if req.window_broken:
                        field.delayed_operations.append((req.operation.operation_id, req.days_delayed))
                    if req.operation.task_type == TaskType.HARVEST:
                        field.finalize_harvest()
                    field.completed_operations.add((req.operation.operation_id, field.phase.value))
                    # запись в Gantt
                    model.gantt_records.append({
                        'operation_name': req.operation.name,
                        'field_id': req.field_id,
                        'start_day': req.start_day if req.start_day is not None else day,
                        'end_day': req.end_day,
                        'required_machines': len(req.assigned_machines),
                        'labor_hours': req.operation.duration_per_ha * field.area,
                        'worker_count': len(req.workers_assigned),
                        'cost': req.total_fuel_cost + req.total_material_cost + req.total_labor_cost,
                        'status': 'completed',
                        'machine_type': req.operation.machine_type.value,
                        'planned_window_end': req.window_end 
                    })
                    # логирование затрат
                    if req.total_water_used > 0:
                        model.log_event(field.unique_id, "Расход воды", f"всего {req.total_water_used:.1f} м³")
                    if req.total_fuel_cost > 0:
                        fuel_liters = req.total_fuel_cost / model.market.get_price('fuel') if model.market.get_price('fuel') > 0 else 0
                        model.log_event(field.unique_id, "Расход топлива", f"всего {fuel_liters:.1f} л")
                    if req.total_labor_cost > 0:
                        labor_hours = req.total_labor_cost / 300
                        model.log_event(field.unique_id, "Трудозатраты", f"всего {labor_hours:.1f} чел·ч")
                    if req.total_material_cost > 0 and req.operation.material_type:
                        price = model.market.get_price(req.operation.material_type) if req.operation.material_type else 0
                        material_units = req.total_material_cost / price if price > 0 else 0
                        model.log_event(field.unique_id, f"Расход {req.operation.material_type}", f"всего {material_units:.1f} кг")
                    model.log_event(field.unique_id, "Завершение операции",
                                    f"операция={req.operation.name}, фактическая длит.={(day - req.start_day) if req.start_day else 0} дн.")
                self.active_requests.remove(req)

        self.active_requests.sort(key=lambda r: (-r.priority, r.window_end - day))

        # Назначение на сегодня
        for req in self.active_requests:
            if req.scheduled_today:
                continue
            if req.remaining_area <= 0:
                continue
            field = model.get_field(req.field_id)
            if not field:
                continue
            if field.is_busy and field.current_request != req:
                continue

            op = req.operation

            # Проверка влажности
            if op.max_soil_moisture and field.soil_moisture > op.max_soil_moisture:
                model.log_event(field.unique_id, "Отказ",
                                f"причина=SOIL_MOISTURE, операция={op.name}, влажность={field.soil_moisture:.1f}, порог={op.max_soil_moisture}")
                continue

            # Дедлайн
            if day > req.window_end and not req.window_broken:
                req.window_broken = True
                req.days_delayed = day - req.window_end
                model.log_event(field.unique_id, "Окно нарушено", f"операция={op.name}, сдвиг={req.days_delayed} дн.")
            if req.window_broken and (day - req.window_end) > 30:
                field.missed_operations.append(req.operation.operation_id)
                field.is_busy = False
                field.current_request = None
                for w in req.workers_assigned:
                    w.release()
                self.active_requests.remove(req)
                model.log_event(field.unique_id, "Операция отменена",
                                f"операция={op.name}, задержка={day - req.window_end} дн., превышен лимит")
                continue

            if req.window_broken:
                days_left = 1
            else:
                days_left = req.window_end - day + 1
            if days_left <= 0:
                continue

            # Производительность одной машины
            machines_of_type = model.machines.get(op.machine_type, [])
            if machines_of_type:
                productivity = machines_of_type[0].productivity
            else:
                productivity = 5.0

            work_left = req.remaining_area
            if not req.operation_started and days_left > 0:
                min_machines = math.ceil(work_left / (productivity * days_left))
                min_machines = max(1, min(min_machines, model.MAX_MACHINES_PER_FIELD))
            else:
                current_machines = len(req.assigned_machines) if req.assigned_machines else 1
                possible_work = current_machines * productivity * days_left
                if possible_work >= work_left - 1e-6:
                    min_machines = current_machines
                else:
                    min_machines = math.ceil(work_left / (productivity * days_left))
                    min_machines = max(current_machines, min(min_machines, model.MAX_MACHINES_PER_FIELD))

            available = [m for m in model.machines.get(op.machine_type, []) if m.status == MachineStatus.IDLE]
            n_avail = len(available)
            n_assign = min(min_machines, n_avail)

            # Адаптация техники
            if min_machines > n_avail and self.enable_adaptation:
                current_total = len(model.machines.get(op.machine_type, []))
                max_total = model.max_machines.get(op.machine_type, float('inf'))
                if current_total < max_total:
                    deficit = min_machines - n_avail
                    can_add = min(deficit, max_total - current_total)
                    if can_add > 0:
                        model.add_machines(op.machine_type, can_add, op, field.unique_id, min_machines, n_avail)
                        available = [m for m in model.machines.get(op.machine_type, []) if m.status == MachineStatus.IDLE]
                        n_avail = len(available)
            n_assign = min(min_machines, n_avail)
            if n_assign == 0:
                model.log_event(field.unique_id, "Отказ",
                                f"причина=NO_MACHINES, операция={op.name}, требуется={min_machines}, доступно=0")
                continue

            daily_work = min(n_assign * productivity, work_left)

            required_hours = daily_work * op.duration_per_ha
            if op.machine_type == MachineType.TRACTOR:
                qual = "tractor_driver"
            elif op.machine_type == MachineType.HARVESTER:
                qual = "harvester_operator"
            else:
                qual = "general"
            free_workers = [w for w in model.workers.get(qual, []) if w.status == "idle"]

            available_hours = sum(w.max_hours_per_day for w in free_workers)

            if required_hours > available_hours:
                if self.enable_adaptation:
                    deficit_hours = required_hours - available_hours
                    # сколько рабочих нужно добавить
                    avg_hours_per_worker = 8.0
                    needed_workers = math.ceil(deficit_hours / avg_hours_per_worker)
                    model.add_workers(qual, needed_workers, op, field.unique_id, required_hours, available_hours)
                    free_workers = [w for w in model.workers.get(qual, []) if w.status == "idle"]
                    available_hours = sum(w.max_hours_per_day for w in free_workers)
                    if required_hours > available_hours:
                        model.log_event(field.unique_id, "Отказ",
                                        f"причина=NOT_ENOUGH_LABOR_HOURS, операция={op.name}, требуется={required_hours:.1f} ч, доступно={available_hours:.1f} ч")
                        continue
                else:
                    model.log_event(field.unique_id, "Отказ",
                                    f"причина=NOT_ENOUGH_LABOR_HOURS, операция={op.name}, требуется={required_hours:.1f} ч, доступно={available_hours:.1f} ч")
                    continue

            # Назначаем рабочих на сегодня
            assigned_workers = []
            remaining_hours = required_hours
            for w in free_workers:
                if remaining_hours <= 0:
                    break
                take_hours = min(w.max_hours_per_day, remaining_hours)
                if w.assign(take_hours, None): 
                    assigned_workers.append(w)
                    remaining_hours -= take_hours
                if remaining_hours <= 0:
                    break
            if remaining_hours > 1e-6:
                model.log_event(field.unique_id, "Отказ",
                                f"причина=LABOR_CAPACITY, операция={op.name}, нехватка {remaining_hours:.1f} ч")
                continue

            # проверка воды
            water_needed = 0
            if op.task_type == TaskType.IRRIGATE:
                water_needed = daily_work * (field.crop.irrigation_amount * 10.0)
                if water_needed > model.remaining_water:
                    if self.enable_adaptation:
                        deficit = water_needed - model.remaining_water
                        model.adjust_water_limit(deficit, op, field.unique_id, water_needed, model.remaining_water)
                    if water_needed > model.remaining_water:
                        model.log_event(field.unique_id, "Отказ",
                                        f"причина=WATER_DEFICIT, операция={op.name}, требуется={water_needed:.1f}, осталось={model.remaining_water:.1f}")
                        continue

            # бюджета
            fuel_price = model.market.get_price('fuel')
            fuel_cost = op.fuel_per_ha * daily_work * fuel_price
            if op.material_type:
                material_price = model.market.get_price(op.material_type)
            else:
                material_price = 0
            material_cost = op.material_consumption * daily_work * material_price

            labor_cost = sum(w.cost_per_hour * op.duration_per_ha * daily_work for w in assigned_workers)
            total_cost = fuel_cost + material_cost + labor_cost
            if total_cost > model.remaining_budget:
                if self.enable_adaptation:
                    deficit = total_cost - model.remaining_budget
                    model.adjust_budget_limit(deficit, op, field.unique_id, total_cost, model.remaining_budget)
                if total_cost > model.remaining_budget:
                    model.log_event(field.unique_id, "Отказ",
                                    f"причина=BUDGET_DEFICIT, операция={op.name}, требуется={total_cost:.0f}, осталось={model.remaining_budget:.0f}")
                    continue

            # Назначение на сегодня
            days_needed = math.ceil(work_left / (n_assign * productivity))
            req.total_fuel_cost += fuel_cost
            req.total_material_cost += material_cost
            req.total_labor_cost += labor_cost
            if op.task_type == TaskType.IRRIGATE:
                req.total_water_used += water_needed
            if not req.operation_started:
                req.workers_needed = len(assigned_workers)
                req.days_needed = days_needed
            req.remaining_area -= daily_work
            if req.start_day is None:
                field.is_busy = True
                field.current_request = req
                req.start_day = day
            req.assigned_machines = available[:n_assign]
            req.workers_assigned = assigned_workers
            req.daily_work = daily_work

            if not req.operation_started:
                req.operation_started = True
                model.log_event(field.unique_id, "Начало операции",
                                f"операция={op.name}, машин={n_assign}, рабочих={len(assigned_workers)}, план. длит.={days_needed} дн.")
            for machine in req.assigned_machines:
                machine.assign_task(field.unique_id, op, field.area, days_needed=days_needed)
            for w in assigned_workers:
                if req.assigned_machines:
                    w.assigned_machine = req.assigned_machines[0].unique_id
            if op.task_type == TaskType.IRRIGATE:
                model.remaining_water -= water_needed
            model.remaining_budget -= total_cost
            field.apply_operation(op, amount=None, fuel_cost=fuel_cost, material_cost=material_cost, labor_cost=labor_cost, water_used=water_needed)
            req.scheduled_today = True

class FarmModel(Model):
    MAX_MACHINES_PER_FIELD = 20
    def __init__(self,
                 fields_data: List[dict],
                 crops: Dict[str, Crop],
                 soils: Dict[str, Soil],
                 machines_data: List[dict],
                 workers_data: List[dict],
                 storages_data: List[dict],
                 operations_data: Dict[int, Operation],
                 tech_cards_data: List[TechCard],
                 weather_daily: List[dict] = None,
                 scenario_probs: Dict[str, float] = None,
                 fixed_areas: Dict[str, float] = None,
                 initial_water_limit: float = 50000,
                 initial_budget: float = 5_000_000,
                 constant_costs: float = 0.0,
                 enable_adaptation: bool = True,
                 max_machines: Dict[MachineType, int] = None,
                 max_workers: Dict[str, int] = None,
                 resource_prices: Dict[str, float] = None,
                 economic_params: Dict[str, float] = None,
                 risks_df: pd.DataFrame = None):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.time = 0
        self.day_of_year = 0
        self.start_date = datetime(2024, 1, 1)
        self.event_log = []
        self.weather_daily = weather_daily
        self.scenario_probs = scenario_probs or {"normal": 1.0}
        self.fixed_areas = fixed_areas or {}
        self.crops = crops
        self.soils = soils
        self.operations = operations_data
        self.tech_cards = tech_cards_data
        self.constant_costs = constant_costs  
        self.water_limit = initial_water_limit
        self.remaining_water = initial_water_limit
        self.budget_limit = initial_budget
        self.remaining_budget = initial_budget
        self.machines = {mt: [] for mt in MachineType}
        self.workers = {}
        self.max_machines = max_machines or {}
        self.max_workers = max_workers or {}
        self._init_machines(machines_data)
        self._init_workers(workers_data)
        self.market = MarketAgent(9998, self, resource_prices)
        self.gantt_records = []
        self.busy_fields = set()
        self.total_variable_cost = 0.0
        self.enable_adaptation = enable_adaptation
        self.adapted_water = set()
        self.adapted_budget = set()
        self.added_machines = set()
        self.added_workers = set()
        self.peak_machines = {mt: 0 for mt in MachineType}
        self.peak_workers = {}
        self.actual_operation_windows = []
        self.adaptation_events = []
        self.total_variable_cost_inc = 0.0

        self.realization_rate = economic_params.get('realization_rate', 0.95) if economic_params else 0.95
        self.risks = []
        self.active_risks = {}
        if risks_df is not None:
            self.load_risks(risks_df)

        self.total_storage_cost = 0.0
        self.last_financial_check_day = 0
        self.economic_params = economic_params or {
            'depreciation_rate_machines': 0.10, 
            'depreciation_rate_perennials': 0.05,
            'property_tax_rate': 0.022, 
            'profit_tax_rate': 0.20,
            'admin_cost_fixed': 500_000.0,
        }
        self.total_machine_initial_cost = sum(m.initial_cost for m in self.machines.values() for m in m)
        self.total_vineyard_cost = 0.0
        self.total_cadastral_value = 0.0
        #стоимость мног культур и кадастровую стоимть полей
        for f in fields_data:
            self.total_cadastral_value += f.get('cadastral_value', 0.0)
            crop_name = f.get('crop_plan')
            if crop_name and crop_name in crops:
                crop = crops[crop_name]
                self.total_vineyard_cost += f['area'] * crop.establishment_cost

        self._ops_by_crop_phase = {}
        for tc in self.tech_cards:
            key = (tc.crop_id, tc.phase_enum)
            self._ops_by_crop_phase.setdefault(key, []).append(tc)

        for f in fields_data:
            soil = self.soils.get(f['soil_name'])
            if not soil:
                continue
            crop_name = f.get('crop_plan')
            if self.fixed_areas and crop_name not in self.fixed_areas:
                crop_name = None
            crop = self.crops.get(crop_name) if crop_name else None
            field = FieldAgent(f['field_id'], self, f, soil, crop)
            self.schedule.add(field)

        self.weather = WeatherAgent(9999, self, weather_daily)
        self.schedule.add(self.weather)
        self.dispatcher = DispatcherAgent(10000, self, enable_adaptation=self.enable_adaptation)
        self.schedule.add(self.dispatcher)

        # Добавляем хранилища
        for st in storages_data:
            store = StorageAgent(st['storage_id'], self, st)
            self.schedule.add(store)

        self.datacollector = DataCollector(
            model_reporters={
                "Day": lambda m: m.day_of_year,
                "Temperature": lambda m: m.weather.temperature,
                "Rain": lambda m: m.weather.precipitation,
                "PendingRequests": lambda m: len(m.dispatcher.active_requests),
                "RemainingWater": lambda m: m.remaining_water,
                "RemainingBudget": lambda m: m.remaining_budget,
            },
            agent_reporters={
                "Phase": lambda a: a.phase.value if isinstance(a, FieldAgent) else None,
                "SoilMoisture": lambda a: a.soil_moisture if isinstance(a, FieldAgent) else None,
                "CropType": lambda a: a.crop_type if isinstance(a, FieldAgent) else None,
                "GDD": lambda a: a.gdd_accumulated if isinstance(a, FieldAgent) else None,
                "MachineStatus": lambda a: a.status.value if isinstance(a, MachineAgent) else None,
                "MachineType": lambda a: a.machine_type.value if isinstance(a, MachineAgent) else None,
            }
        )

    def _init_machines(self, machines_data):
        for m in machines_data:
            machine = MachineAgent(m['machine_id'], self, m)
            self.machines[machine.machine_type].append(machine)
            self.schedule.add(machine)

    def _init_workers(self, workers_data):
        for w in workers_data:
            worker = WorkerAgent(w['worker_id'], self, w)
            self.workers.setdefault(worker.qualification, []).append(worker)
            self.schedule.add(worker)

    def add_machines(self, machine_type: MachineType, count: int, operation=None, field_id=None, required=None, available=None):
        current = len(self.machines.get(machine_type, []))
        max_allowed = self.max_machines.get(machine_type, float('inf'))
        if current >= max_allowed:
            self.log_event(-1, "Адаптация отклонена", f"Достигнут лимит техники {machine_type.value}: {current}/{max_allowed}")
            return
        count = min(count, max_allowed - current)
        if count <= 0:
            return
        for _ in range(count):
            new_id = max([a.unique_id for a in self.schedule.agents] + [0]) + 1
            new_machine = MachineAgent(new_id, self, {
                'machine_id': new_id,
                'type': machine_type.value,
                'brand': 'Adaptive',
                'productivity': 5,
                'fuel_consumption': 10,
                'fuel_capacity': 200,
                'cost_hour': 1000,
                'requires_operator': True,
                'initial_cost': 500_000,   # условно
                'useful_life_years': 10
            })
            self.machines.setdefault(machine_type, []).append(new_machine)
            self.schedule.add(new_machine)
        desc = f"Добавлена техника {machine_type.value} в количестве {count} (операция {operation.name if operation else '?'}, поле {field_id}, требовалось {required}, доступно {available})"
        self.log_event(-1, "Адаптация", desc)
        self.adaptation_events.append({
            'day': self.day_of_year,
            'type': 'add_machines',
            'resource': machine_type.value,
            'amount': count,
            'operation': operation.name if operation else '?',
            'field_id': field_id,
            'required': required,
            'available': available
        })

    def load_risks(self, risks_df: pd.DataFrame):
        self.risks = [Risk(row.to_dict()) for _, row in risks_df.iterrows()]

    def add_workers(self, qualification: str, count: int, operation=None, field_id=None, required=None, available=None):
        
        current = len(self.workers.get(qualification, []))
        max_allowed = self.max_workers.get(qualification, float('inf'))
        if current >= max_allowed:
            self.log_event(-1, "Адаптация отклонена", f"Достигнут лимит рабочих {qualification}: {current}/{max_allowed}")
            return
        count = min(count, max_allowed - current)
        if count <= 0:
            return
        for _ in range(count):
            new_id = max([a.unique_id for a in self.schedule.agents] + [0]) + 1
            new_worker = WorkerAgent(new_id, self, {
                'worker_id': new_id,
                'name': f'Worker_{new_id}',
                'qualification': qualification,
                'cost_per_hour': 300,
                'max_hours_per_day': 8
            })
            self.workers.setdefault(qualification, []).append(new_worker)
            self.schedule.add(new_worker)
        desc = f"Добавлен рабочий {qualification} в количестве {count} (операция {operation.name if operation else '?'}, поле {field_id}, требовалось {required}, доступно {available})"
        self.log_event(-1, "Адаптация", desc)
        self.adaptation_events.append({
            'day': self.day_of_year,
            'type': 'add_workers',
            'resource': qualification,
            'amount': count,
            'operation': operation.name if operation else '?',
            'field_id': field_id,
            'required': required,
            'available': available
        })

    def adjust_water_limit(self, deficit: float, operation=None, field_id=None, required=None, available=None):
        old_limit = self.water_limit
        self.water_limit += deficit
        self.remaining_water += deficit
        desc = f"Увеличен лимит-вода на {deficit:.1f} м³ (операция {operation.name if operation else '?'}, поле {field_id}, требовалось {required:.1f}, доступно {available:.1f})"
        self.log_event(-1, "Адаптация", desc)
        self.adaptation_events.append({
            'day': self.day_of_year,
            'type': 'adjust_water_limit',
            'resource': 'water',
            'amount': deficit,
            'operation': operation.name if operation else '?',
            'field_id': field_id,
            'required': required,
            'available': available,
            'old_limit': old_limit,
            'new_limit': self.water_limit
        })

    def adjust_budget_limit(self, deficit: float, operation=None, field_id=None, required=None, available=None):
        old_limit = self.budget_limit
        self.budget_limit += deficit
        self.remaining_budget += deficit
        desc = f"Увеличен бюджет на {deficit:.0f} руб (операция {operation.name if operation else '?'}, поле {field_id}, требовалось {required:.0f}, доступно {available:.0f})"
        self.log_event(-1, "Адаптация", desc)
        self.adaptation_events.append({
            'day': self.day_of_year,
            'type': 'adjust_budget_limit',
            'resource': 'budget',
            'amount': deficit,
            'operation': operation.name if operation else '?',
            'field_id': field_id,
            'required': required,
            'available': available,
            'old_limit': old_limit,
            'new_limit': self.budget_limit
        })

    def compute_constant_costs(self) -> Dict[str, float]:
        depr_machines = 0.0
        for mt, machines in self.machines.items():
            for m in machines:
                depr_machines += m.initial_cost * self.economic_params['depreciation_rate_machines']
        depr_perennials = self.total_vineyard_cost * self.economic_params['depreciation_rate_perennials']
        residual_machines = max(0.0, self.total_machine_initial_cost - depr_machines)
        residual_vineyard = max(0.0, self.total_vineyard_cost - depr_perennials)
        property_tax = (residual_machines + residual_vineyard + self.total_cadastral_value) * self.economic_params['property_tax_rate']
        return {
            'depreciation': depr_machines + depr_perennials,
            'property_tax': property_tax
        }

    def compute_financial_metrics(self, revenue: float, total_variable_cost: float, extra_risk_cost: float = 0.0) -> Dict[str, float]:
        admin_costs = self.economic_params['admin_cost_fixed']
        ebitda = revenue - total_variable_cost - self.total_storage_cost - admin_costs
        constant = self.compute_constant_costs()
        profit_before_tax = ebitda - constant['depreciation'] - constant['property_tax']
        income_tax = max(0.0, profit_before_tax * self.economic_params['profit_tax_rate'])
        net_profit = profit_before_tax - income_tax
        return {
            'revenue': revenue,
            'total_variable_cost': total_variable_cost,
            'extra_risk_cost': extra_risk_cost,
            'storage_cost': self.total_storage_cost,
            'admin_cost': admin_costs,
            'ebitda': ebitda,
            'depreciation': constant['depreciation'],
            'property_tax': constant['property_tax'],
            'profit_before_tax': profit_before_tax,
            'income_tax': income_tax,
            'net_profit': net_profit
        }

    def get_delayed_operations(self) -> List[Dict]:
        delayed = []
        for agent in self.schedule.agents:
            if isinstance(agent, FieldAgent):
                for op_id, days in agent.delayed_operations:
                    delayed.append({
                        'field_id': agent.unique_id,
                        'operation_id': op_id,
                        'delay_days': days
                    })
        return delayed
    
    def get_canceled_operations(self) -> List[Dict]:
        canceled = []
        for agent in self.schedule.agents:
            if isinstance(agent, FieldAgent):
                for op_id in agent.missed_operations:
                    op = self.get_operation_by_id(op_id)
                    canceled.append({
                        'field_id': agent.unique_id,
                        'operation_id': op_id,
                        'operation_name': op.name if op else str(op_id),
                        'reason': 'Превышена допустимая задержка (>20 дней)'
                    })
        return canceled
    
    def get_free_machine(self, machine_type: MachineType, create_if_missing=False) -> Optional[MachineAgent]:
        for m in self.machines.get(machine_type, []):
            if m.status == MachineStatus.IDLE:
                return m
        if create_if_missing and self.enable_adaptation:
            self.add_machines(machine_type, 1)
            return self.get_free_machine(machine_type, create_if_missing=False)
        return None

    def get_free_workers(self, qualification: str, needed: int) -> List[WorkerAgent]:
        free = [w for w in self.workers.get(qualification, []) if w.status == "idle"]
        return free[:needed]

    def assign_worker(self, machine_id: int, machine_type: MachineType):
        if machine_type == MachineType.TRACTOR:
            qual = "tractor_driver"
        elif machine_type == MachineType.HARVESTER:
            qual = "harvester_operator"
        else:
            qual = "general"
        worker = self.get_free_workers(qual, 1)
        if worker:
            w = worker[0]
            w.status = "busy"
            w.assigned_machine = machine_id

    def release_worker(self, machine_id: int):
        for quals, workers in self.workers.items():
            for w in workers:
                if w.assigned_machine == machine_id:
                    w.status = "idle"
                    w.assigned_machine = None
                    return

    def get_operation_by_id(self, op_id: int) -> Optional[Operation]:
        return self.operations.get(op_id)

    def get_operation_by_type(self, task_type: TaskType) -> Optional[Operation]:
        for op in self.operations.values():
            if op.task_type == task_type:
                return op
        return None

    def get_operations_for_crop_phase(self, crop_name: str, phase: str) -> List[TechCard]:
        crop = self.crops.get(crop_name)
        if not crop:
            return []
        crop_id = crop.crop_id
        return self._ops_by_crop_phase.get((crop_id, phase), [])

    def add_request(self, field_id: int, operation: Operation, priority: int, window_end: int = None, total_area: float = None):
        if total_area is None:
            field = self.get_field(field_id)
            if field:
                total_area = field.area
            else:
                total_area = 0
        self.dispatcher.add_request(field_id, operation, priority, window_end, total_area)
        self.log_event(field_id, "Заявка добавлена", f"Операция: {operation.name}, приоритет {priority}")

    def notify_operation_complete(self, field_id: int, machine: MachineAgent):
        pass

    def get_field(self, field_id: int) -> Optional[FieldAgent]:
        for agent in self.schedule.agents:
            if isinstance(agent, FieldAgent) and agent.unique_id == field_id:
                return agent
        return None

    def get_storage_for(self, storage_type: str) -> Optional[StorageAgent]:
        for agent in self.schedule.agents:
            if isinstance(agent, StorageAgent) and agent.storage_type == storage_type:
                return agent
        return None

    def get_missed_operations(self) -> List[Dict]:
        missed = []
        for event in self.event_log:
            if event['event_type'] == "Отказ" and "WINDOW_CLOSED" in event['description']:
                missed.append({
                    'day': event['day'],
                    'field_id': event['agent_id'],
                    'operation': event['description'].split('операция=')[1].split(',')[0] if 'операция=' in event['description'] else "Неизвестно",
                    'reason': 'Окно закрыто'
                })
        return missed
    
    def cleanup_field_requests(self, field_id):
        for req in self.dispatcher.active_requests[:]:
            if req.field_id == field_id and req.remaining_area > 0:
                field = self.get_field(field_id)
                if field:
                    field.missed_operations.append(req.operation.operation_id)
                if req.window_broken:
                    field.delayed_operations.append((req.operation.operation_id, req.days_delayed))
                self.dispatcher.active_requests.remove(req)
    
    def step(self):
        self.time += 1
        self.day_of_year += 1
        for qual, workers_list in self.workers.items():
            for w in workers_list:
                if self.day_of_year != w.last_day:
                    w.hours_worked_today = 0
                    w.last_day = self.day_of_year
                    w.status = "idle"
                    w.assigned_machine = None
        self.schedule.step()
        self.datacollector.collect(self)

        for mt in MachineType:
            busy_count = sum(1 for m in self.machines.get(mt, []) if m.status == MachineStatus.BUSY)
            if busy_count > self.peak_machines[mt]:
                self.peak_machines[mt] = busy_count
        for qual, workers in self.workers.items():
            busy_count = sum(1 for w in workers if w.status == "busy")
            current_peak = self.peak_workers.get(qual, 0)
            if busy_count > current_peak:
                self.peak_workers[qual] = busy_count
        
        # обновление рисков
        for field_id, risks in list(self.active_risks.items()):
            new_risks = []
            for risk, days in risks:
                if days > 1:
                    new_risks.append((risk, days-1))
            if new_risks:
                self.active_risks[field_id] = new_risks
            else:
                del self.active_risks[field_id]

    def log_event(self, agent_id: int, event_type: str, description: str):
        if self.event_log:
            last = self.event_log[-1]
            if last['day'] == self.day_of_year and last['event_type'] == event_type and last['description'] == description:
                return
        self.event_log.append({
            "day": self.day_of_year,
            "date": (self.start_date + timedelta(days=self.day_of_year-1)).strftime("%Y-%m-%d"),
            "agent_id": agent_id,
            "event_type": event_type,
            "description": description
        })

    def compute_actual_profit(self) -> Tuple[float, float]:
        total_revenue = 0.0
        total_variable_cost = 0.0
        for agent in self.schedule.agents:
            if isinstance(agent, FieldAgent) and agent.crop:
                yld = agent.get_yield()
                if yld > 0:
                    revenue = yld * agent.crop.price
                    total_revenue += revenue
                    seed_cost = agent.area * agent.crop.seed_cost
                    
                    water_cost = agent.total_water_used * self.market.get_price('water')
                    fuel_cost = agent.total_fuel_used * self.market.get_price('fuel')
                    
                    material_cost = agent.total_material_used
                    labor_cost = agent.total_labor_cost
                    total_variable_cost += seed_cost + water_cost + fuel_cost + material_cost + labor_cost
        total_variable_cost += self.total_variable_cost_inc 
        revenue = total_revenue * self.realization_rate
        profit = revenue - total_variable_cost - self.constant_costs
        return revenue, profit

    def get_crop_yield(self, crop_name: str) -> float:
        total_yield = 0.0
        total_area = 0.0
        for agent in self.schedule.agents:
            if isinstance(agent, FieldAgent) and agent.crop and agent.crop.name == crop_name:
                total_yield += agent.get_yield()
                total_area += agent.area
        return total_yield / total_area if total_area > 0 else 0.0

    def get_final_limits(self) -> Dict:
        return {
            'water_limit': self.water_limit,
            'budget_limit': self.budget_limit,
            'machines': {mt.value: len(lst) for mt, lst in self.machines.items()},
            'workers': {qual: len(lst) for qual, lst in self.workers.items()},
            'peak_machines': {mt.value: self.peak_machines[mt] for mt in MachineType},
            'peak_workers': self.peak_workers.copy(),
            'actual_operation_windows': self.actual_operation_windows,
            'adaptation_events': self.adaptation_events,
            'canceled_operations': self.get_canceled_operations(),
            'delayed_operations': self.get_delayed_operations(),
            'water_usage_by_field':self.get_water_usage_by_field().to_dict('records')
        }

    def reduce_limits_to_peak(self):
        for mt, peak in self.peak_machines.items():
            current = len(self.machines.get(mt, []))
            if current > peak:
                excess = current - peak
                for _ in range(excess):
                    machine = self.machines[mt].pop()
                    self.schedule.remove(machine)
        for qual, peak in self.peak_workers.items():
            current = len(self.workers.get(qual, []))
            if current > peak:
                excess = current - peak
                for _ in range(excess):
                    worker = self.workers[qual].pop()
                    self.schedule.remove(worker)
    
    def get_water_usage_by_field(self) -> pd.DataFrame:
        data = []
        for agent in self.schedule.agents:
            if isinstance(agent, FieldAgent) and agent.crop:
                water_total = agent.total_water_used  
                area = agent.area
                water_per_ha = water_total / area if area > 0 else 0
                data.append({
                'Поле ID': agent.unique_id,
                'Культура': agent.crop.name,
                'Общее водопотребление (м³)': round(water_total, 1),
                'Водопотребление на га (м³/га)': round(water_per_ha, 1)
                })
        return pd.DataFrame(data)

def estimate_phase_dates(crop, weather_df: pd.DataFrame, base_temp: float, return_debug=False) -> Dict[str, int]:
    phases_sorted = sorted(
        [p for p in crop.phases.values() if p.gdd_required > 0],
        key=lambda p: p.gdd_required
    )
    if not phases_sorted:
        return {} if not return_debug else ({}, 0, [])
    gdd_cum = 0.0
    phase_dates = {}
    achieved_phases = []
    next_phase_idx = 0
    for day in range(1, 366):
        day_data = weather_df[weather_df['day'] == day]
        if day_data.empty:
            continue
        tmin = day_data['tmin'].values[0]
        tmax = day_data['tmax'].values[0]
        gdd = calc_gdd(tmin, tmax, base_temp)
        gdd_cum += gdd
        while next_phase_idx < len(phases_sorted) and gdd_cum >= phases_sorted[next_phase_idx].gdd_required:
            phase = phases_sorted[next_phase_idx]
            phase_dates[phase.phase_enum] = day
            achieved_phases.append(phase.phase_enum)
            next_phase_idx += 1
    if return_debug:
        return phase_dates, gdd_cum, achieved_phases
    return phase_dates

def temperature_stress_with_params(temp: float, Tmin, Topt1, Topt2, Tmax) -> float:
    if temp < Tmin or temp > Tmax:
        return 0.0
    if temp < Topt1:
        return (temp - Tmin) / (Topt1 - Tmin)
    if temp <= Topt2:
        return 1.0
    return (Tmax - temp) / (Tmax - Topt2)

def precip_stress_with_params(precip: float, Pmin, Popt1, Popt2, Pmax) -> float:
    if precip < Pmin or precip > Pmax:
        return 0.0
    if precip < Popt1:
        return (precip - Pmin) / (Popt1 - Pmin)
    if precip <= Popt2:
        return 1.0
    return (Pmax - precip) / (Pmax - Popt2)

def par_stress_with_params(par: float, PARmin, PARopt1, PARopt2, PARmax) -> float:
    if par < PARmin or par > PARmax:
        return 0.0
    if par < PARopt1:
        return (par - PARmin) / (PARopt1 - PARmin)
    if par <= PARopt2:
        return 1.0
    return (PARmax - par) / (PARmax - PARopt2)

def get_last_phase(crop: Crop) -> Optional[str]:
    for phase_name, phase_info in crop.phases.items():
        if not phase_info.next_phase:
            return phase_name
    return None

def compute_potential_yield(crop: Crop, weather_df: pd.DataFrame, base_temp: float = None, return_debug=False) -> float:
    if base_temp is None:
        base_temp = crop.T_base
    phase_dates, gdd_cum, achieved_phases = estimate_phase_dates(crop, weather_df, base_temp, return_debug=True)
    if not phase_dates:
        if return_debug:
            return crop.yield_pot, f"⚠️ Ни одна фаза не достигнута. Накоплено GDD={gdd_cum:.0f}"
        return crop.yield_pot
    last_phase = get_last_phase(crop)
    last_achieved = achieved_phases[-1] if achieved_phases else None
    warning = None
    if last_phase and last_phase not in achieved_phases:
        required_gdd = crop.phases[last_phase].gdd_required
        warning = f"Финальная фаза '{last_phase}' не достигнута. Последняя: {last_achieved}, GDD={gdd_cum:.0f}, требуется {required_gdd}"
    sorted_phases = sorted(phase_dates.items(), key=lambda x: x[1])
    total_factor = 1.0
    for idx, (phase_enum, start_day) in enumerate(sorted_phases):
        end_day = sorted_phases[idx+1][1] - 1 if idx+1 < len(sorted_phases) else 365
        phase_info = crop.phases.get(phase_enum)
        if not phase_info:
            continue
        mask = (weather_df['day'] >= start_day) & (weather_df['day'] <= end_day)
        if not mask.any():
            continue
        subset = weather_df[mask]
        tmean = (subset['tmin'] + subset['tmax']).mean() / 2.0
        par = subset['par'].mean()
        Tmin = phase_info.Tmin if phase_info.Tmin is not None else crop.T_min
        Topt1 = phase_info.Topt1 if phase_info.Topt1 is not None else crop.T_opt1
        Topt2 = phase_info.Topt2 if phase_info.Topt2 is not None else crop.T_opt2
        Tmax = phase_info.Tmax if phase_info.Tmax is not None else crop.T_max
        PARmin = phase_info.PAR_min if phase_info.PAR_min is not None else 5
        PARopt1 = phase_info.PAR_opt1 if phase_info.PAR_opt1 is not None else 10
        PARopt2 = phase_info.PAR_opt2 if phase_info.PAR_opt2 is not None else 20
        PARmax = phase_info.PAR_max if phase_info.PAR_max is not None else 30
        stress_temp = temperature_stress_with_params(tmean, Tmin, Topt1, Topt2, Tmax)
        stress_par = par_stress_with_params(par, PARmin, PARopt1, PARopt2, PARmax)
        weights = phase_info.weights if phase_info.weights else {"temp":0.5, "par":0.2}
        total_weight = weights.get('temp',0) + weights.get('par',0)
        if total_weight > 0:
            phase_factor = (weights.get('temp',0)*stress_temp + weights.get('par',0)*stress_par) / total_weight
        else:
            phase_factor = 1.0
        total_factor *= phase_factor
    final_yield = crop.yield_pot * total_factor
    if return_debug:
        return final_yield, warning
    return final_yield

def compute_yield_scenarios(crops: Dict[str, Crop], weather_scenarios: Dict[str, pd.DataFrame],
                            base_temps: Dict[str, float] = None) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, str]]]:
    result = {}
    warnings = {}
    for scen_name, wdf in weather_scenarios.items():
        result[scen_name] = {}
        warnings[scen_name] = {}
        for crop_name, crop in crops.items():
            base_temp = base_temps.get(crop_name, crop.T_base) if base_temps else crop.T_base
            y, warn = compute_potential_yield(crop, wdf, base_temp, return_debug=True)
            result[scen_name][crop_name] = y
            if warn:
                warnings[scen_name][crop_name] = warn
    return result, warnings

def compute_period_norms(crops: Dict[str, Crop], tech_cards_df: pd.DataFrame,
                         operations_df: pd.DataFrame, weather_df: pd.DataFrame,
                         num_periods: int = 36) -> Tuple[Dict, Dict]:
    labor_norm = {}
    machine_norm = {}
    crop_id_to_name = {crop.crop_id: name for name, crop in crops.items()}
    crop_phase_dates = {}
    for crop_name, crop in crops.items():
        base_temp = crop.T_base
        phase_dates = estimate_phase_dates(crop, weather_df, base_temp)
        if not phase_dates:
            phase_dates = {'sowing': crop.sowing_start}
        crop_phase_dates[crop_name] = phase_dates
    for _, tc in tech_cards_df.iterrows():
        crop_name = crop_id_to_name.get(tc['crop_id'])
        if not crop_name:
            continue
        crop = crops[crop_name]
        phase_enum = tc.get('phase_enum')
        if not phase_enum or phase_enum not in crop_phase_dates[crop_name]:
            phase_start_abs = crop.sowing_start
        else:
            phase_start_abs = crop_phase_dates[crop_name][phase_enum]
        op_id = tc['operation_id']
        op_row = operations_df[operations_df['operation_id'] == op_id]
        if op_row.empty:
            continue
        op = op_row.iloc[0]
        machine_type = op['machine_type']
        duration_per_ha = op['duration_per_ha']
        day_start_rel = tc['day_start']
        day_end_rel = tc['day_end']
        abs_start = phase_start_abs + day_start_rel
        abs_end = phase_start_abs + day_end_rel
        abs_start = max(1, min(abs_start, 365))
        abs_end = max(1, min(abs_end, 365))
        if abs_start > abs_end:
            continue
        start_period = (abs_start - 1) // 10
        end_period = (abs_end - 1) // 10
        total_days = abs_end - abs_start + 1

        for p in range(start_period, end_period + 1):
            p_start_day = p * 10 + 1
            p_end_day = min((p + 1) * 10, 365)
            overlap_start = max(abs_start, p_start_day)
            overlap_end = min(abs_end, p_end_day)
            if overlap_end < overlap_start:
                continue

            days_in_period = overlap_end - overlap_start + 1
            fraction = days_in_period / total_days
            contrib = fraction * duration_per_ha
            labor_norm[(crop_name, p, 'general')] = labor_norm.get((crop_name, p, 'general'), 0) + contrib
            machine_norm[(crop_name, p, machine_type)] = machine_norm.get((crop_name, p, machine_type), 0) + contrib
    return labor_norm, machine_norm

def compute_full_variable_cost(crop: Crop, operations: Dict[int, Operation], tech_cards: List[TechCard],
                            labor_cost_per_hour: float, resource_prices: Dict[str, float]) -> float:
    total = crop.seed_cost
    for tc in tech_cards:
        if tc.crop_id != crop.crop_id:
            continue
        op = operations.get(tc.operation_id)
        if not op:
            continue
        fuel_cost = op.fuel_per_ha * resource_prices.get('fuel', 60)
        if op.task_type == TaskType.IRRIGATE:
            water_m3_per_ha = op.material_consumption * 10
            material_cost = water_m3_per_ha * resource_prices.get('water', 0)
        else:
            material_cost = op.material_consumption * resource_prices.get(op.material_type, 0)
        labor_cost = op.duration_per_ha * labor_cost_per_hour if op.requires_operator else 0
        total += fuel_cost + material_cost + labor_cost
    return total

def optimize_structure(crops: Dict[str, Crop],
                       total_area: float,
                       water_limit: float,
                       budget_limit: float,
                       scenario_probs: Dict[str, float],
                       yield_scenarios: Dict[str, Dict[str, float]],
                       var_costs: Dict[str, float],
                       labor_norm: Dict[Tuple[str, int, str], float],
                       labor_limits: Dict[Tuple[int, str], float],
                       machine_norm: Dict[Tuple[str, int, str], float],
                       machine_limits: Dict[Tuple[int, str], float],
                       min_areas: Dict[str, float] = None,
                       max_shares: Dict[str, float] = None,
                       periods: List[int] = None) -> Tuple[Dict[str, float], float]:
    prob = pl.LpProblem("Crop_Optimization", pl.LpMaximize)
    crop_names = list(crops.keys())
    x = {c: pl.LpVariable(f"x_{c}", lowBound=0) for c in crop_names}
    expected_profit = 0

    for s, prob_s in scenario_probs.items():
        for c in crop_names:
            yld = yield_scenarios.get(s, {}).get(c, 0)
            price = crops[c].price
            var_cost = var_costs.get(c, 0)
            expected_profit += prob_s * (x[c] * yld * price - x[c] * var_cost)
    prob += expected_profit
    prob += pl.lpSum(x[c] for c in crop_names) <= total_area
    prob += pl.lpSum(x[c] * crops[c].water_need for c in crop_names) <= water_limit
    prob += pl.lpSum(x[c] * var_costs.get(c, 0) for c in crop_names) <= budget_limit

    if min_areas:
        for c, min_a in min_areas.items():
            prob += x[c] >= min_a
    if max_shares:
        for c, max_share in max_shares.items():
            prob += x[c] <= max_share * total_area
    try:
        prob.solve(pl.PULP_CBC_CMD(msg=True))
    except Exception as e:
        print(f"Solver error: {e}")
        raise
    if prob.status != pl.LpStatusOptimal:
        print(f"LP status: {prob.status}")
        raise ValueError(f"LP не имеет решения. Статус: {prob.status}")
    
    result = {c: x[c].varValue for c in crop_names}
    return result, pl.value(prob.objective)

def run_simulation_with_structure(crops: Dict[str, Crop], soils: Dict[str, Soil], fields_data: List[dict],
                                  machines_data: List[dict], workers_data: List[dict], storages_data: List[dict],
                                  operations: Dict[int, Operation], tech_cards: List[TechCard],
                                  weather_daily: List[dict], optimal_areas: Dict[str, float] = None,
                                  days: int = 365, initial_water_limit: float = 50000,
                                  initial_budget: float = 5_000_000, constant_costs: float = 0.0,
                                  enable_adaptation: bool = True,
                                  max_machines: Dict[MachineType, int] = None,
                                  max_workers: Dict[str, int] = None,
                                  resource_prices: Dict[str, float] = None,
                                  economic_params: Dict[str, float] = None,
                                  risks_df: pd.DataFrame = None) -> Tuple[float, float, Dict[str, float], pd.DataFrame, pd.DataFrame, Dict, List, List, Dict]:
    updated_fields = distribute_crops_to_fields(fields_data, optimal_areas)
    if max_machines is None:
        max_machines = {
            MachineType.TRACTOR: 100,
            MachineType.HARVESTER: 100,
            MachineType.IRRIGATOR: 100,
            MachineType.SPRAYER: 100,
            MachineType.TRANSPORTER: 100
        }
    if max_workers is None:
        max_workers = {
            "tractor_driver": 100,
            "harvester_operator": 100,
            "general": 100
        }
    model = FarmModel(
        fields_data=updated_fields,
        crops=crops,
        soils=soils,
        machines_data=machines_data,
        workers_data=workers_data,
        storages_data=storages_data,
        operations_data=operations,
        tech_cards_data=tech_cards,
        weather_daily=weather_daily,
        initial_water_limit=initial_water_limit,
        initial_budget=initial_budget,
        constant_costs=constant_costs,
        enable_adaptation=enable_adaptation,
        max_machines=max_machines,
        max_workers=max_workers,
        resource_prices=resource_prices,
        economic_params=economic_params,
        risks_df=risks_df
    )
    for _ in range(days):
        model.step()
    revenue, profit_old = model.compute_actual_profit() 

    total_variable_cost = 0.0
    for agent in model.schedule.agents:
        if isinstance(agent, FieldAgent) and agent.crop:
            seed_cost = agent.area * agent.crop.seed_cost
            water_cost = agent.total_water_used * model.market.get_price('water')
            fuel_cost = agent.total_fuel_used * model.market.get_price('fuel')
            material_cost = agent.total_material_used
            labor_cost = agent.total_labor_cost
            total_variable_cost += seed_cost + water_cost + fuel_cost + material_cost + labor_cost
    
    extra_risk_cost = model.total_variable_cost_inc
    total_variable_cost += extra_risk_cost
    financial_metrics = model.compute_financial_metrics(revenue, total_variable_cost, extra_risk_cost)
    yields = {}
    for agent in model.schedule.agents:
        if isinstance(agent, FieldAgent) and agent.crop:
            crop_name = agent.crop.name
            y = agent.get_yield() / agent.area if agent.area > 0 else 0
            total_area_crop = sum(f['area'] for f in updated_fields if f.get('crop_plan') == crop_name)
            if total_area_crop > 0:
                yields[crop_name] = yields.get(crop_name, 0) + y * agent.area / total_area_crop
    agent_df = model.datacollector.get_agent_vars_dataframe()
    event_df = pd.DataFrame(model.event_log)
    if event_df.empty:
        event_df = pd.DataFrame(columns=['day', 'date', 'agent_id', 'event_type', 'description'])
    final_limits = model.get_final_limits()
    gantt_records = model.gantt_records
    missed_ops = model.get_missed_operations()

    return financial_metrics['net_profit'], revenue, yields, agent_df, event_df, final_limits, gantt_records, missed_ops, financial_metrics

def distribute_crops_to_fields(fields_data: List[dict], crop_areas: Dict[str, float]) -> List[dict]:
    if crop_areas is None:
        return fields_data
    fields = [f.copy() for f in fields_data]
    remaining = crop_areas.copy()
    for f in fields:
        f['crop_plan'] = None
    fields.sort(key=lambda f: f['area'], reverse=True)
    for crop, need in sorted(remaining.items(), key=lambda x: -x[1]):
        assigned = False
        for f in fields:
            if f['crop_plan'] is None and f['area'] >= need - 0.1:
                f['crop_plan'] = crop
                assigned = True
                break
        if not assigned:
            for f in fields:
                if f['crop_plan'] is None:
                    f['crop_plan'] = crop
                    assigned = True
                    break
        if assigned:
            pass
    return fields

def adaptive_resource_optimization(
    crops: Dict[str, Crop],
    total_area: float,
    initial_water_limit: float,
    initial_budget: float,
    initial_labor_limits: Dict[Tuple[int, str], float],
    initial_machine_limits: Dict[Tuple[int, str], float],
    yield_scenarios: Dict[str, Dict[str, float]],
    scenario_probs: Dict[str, float],
    min_areas: Dict[str, float],
    max_shares: Dict[str, float],
    fields_data: List[dict],
    soils: Dict[str, Soil],
    machines_data: List[dict],
    workers_data: List[dict],
    storages_data: List[dict],
    operations: Dict[int, Operation],
    tech_cards: List[TechCard],
    weather_daily_by_scenario: Dict[str, pd.DataFrame],
    var_costs: Dict[str, float] = None,
    optimal_areas: Dict[str, float] = None,
    constant_costs: float = 0.0,
    enable_adaptation: bool = True,
    resource_prices: Dict[str, float] = None,
    economic_params: Dict[str, float] = None,
    risks_df: pd.DataFrame = None
) -> Dict:
    
    yield_scenarios, phenology_warnings = compute_yield_scenarios(crops, weather_daily_by_scenario)
    
    if optimal_areas is None:
        optimal_areas = {}
        for f in fields_data:
            crop = f.get('crop_plan')
            if crop:
                optimal_areas[crop] = optimal_areas.get(crop, 0) + f['area']
        lp_profit_val = 0.0
    else:
        if yield_scenarios is None or scenario_probs is None:
            lp_profit_val = 0.0
        else:
            try:
                base_scen = list(weather_daily_by_scenario.keys())[0]
                base_weather_df = weather_daily_by_scenario[base_scen]
                tech_cards_df = pd.DataFrame([tc.__dict__ for tc in tech_cards])
                operations_df = pd.DataFrame([op.__dict__ for op in operations.values()])
                labor_norm, machine_norm = compute_period_norms(crops, tech_cards_df, operations_df, base_weather_df, 36)
                labor_limits = initial_labor_limits if initial_labor_limits else {}
                machine_limits = initial_machine_limits if initial_machine_limits else {}
                periods = list(range(36))
                lp_result, lp_profit_val = optimize_structure(
                    crops=crops,
                    total_area=total_area,
                    water_limit=initial_water_limit,
                    budget_limit=initial_budget,
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
                optimal_areas = lp_result
            except Exception as e:
                print(f"LP failed: {e}")
                lp_profit_val = 0.0
    scenario_results = {}
    max_water = initial_water_limit
    max_budget = initial_budget
    max_machines = {}
    max_workers = {}
    for scen_name, weather_df in weather_daily_by_scenario.items():
        weather_daily = weather_df.to_dict('records')
        profit, revenue, yields, agent_df, event_df, final_limits, gantt, missed, financial_metrics = run_simulation_with_structure(
            crops=crops,
            soils=soils,
            fields_data=fields_data,
            machines_data=machines_data,
            workers_data=workers_data,
            storages_data=storages_data,
            operations=operations,
            tech_cards=tech_cards,
            weather_daily=weather_daily,
            optimal_areas=optimal_areas,
            days=365,
            initial_water_limit=initial_water_limit,
            initial_budget=initial_budget,
            constant_costs=constant_costs,
            enable_adaptation=enable_adaptation,
            resource_prices=resource_prices,
            economic_params=economic_params,
            risks_df=risks_df
        )
        scenario_results[scen_name] = {
            'profit': profit,
            'revenue': revenue,
            'yields': yields,
            'agent_df': agent_df,
            'event_df': event_df,
            'limits': final_limits,
            'gantt': gantt,
            'missed_operations': missed,
            'financial_metrics': financial_metrics
        }
        max_water = max(max_water, final_limits['water_limit'])
        max_budget = max(max_budget, final_limits['budget_limit'])
        for mt, count in final_limits.get('machines', {}).items():
            max_machines[mt] = max(max_machines.get(mt, 0), count)
        for qual, count in final_limits.get('workers', {}).items():
            max_workers[qual] = max(max_workers.get(qual, 0), count)
    return {
        'optimal_areas': optimal_areas,
        'lp_profit': lp_profit_val,
        'adjusted_water_limit': max_water,
        'adjusted_budget': max_budget,
        'adjusted_machines': max_machines,
        'adjusted_workers': max_workers,
        'scenario_results': scenario_results,
        'phenology_warnings': phenology_warnings
    }

def plot_gdd_and_phases(agent_df: pd.DataFrame, field_id: int, crop: Crop = None) -> go.Figure:
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

    data = flat[flat['AgentID'] == field_id].copy()
    if data.empty:
        return go.Figure()
    if 'GDD' not in data.columns or 'Phase' not in data.columns:
        return go.Figure()

    data = data.groupby('Step', as_index=False).agg({
        'GDD': 'first',
        'Phase': 'first'
    }).sort_values('Step')
    data = data[data['Phase'].notna() & (data['Phase'] != '')]
    if data.empty:
        return go.Figure()

    phase_changes = []
    prev_phase = None
    for _, row in data.iterrows():
        curr = row['Phase']
        if curr != prev_phase and prev_phase is not None:
            phase_changes.append((row['Step'], prev_phase, curr))
        prev_phase = curr

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Step'], y=data['GDD'],
        mode='lines', name='Накопленная GDD',
        line=dict(color='blue', width=2)
    ))

    for day, old, new in phase_changes:
        fig.add_vline(x=day, line_dash="dash", line_color="gray", opacity=0.6)
        if old == "SOWING":
            fig.add_annotation(x=day, y=data['GDD'].max() * 0.1, text="Начало вегетации",
                               showarrow=True, arrowhead=1, ax=30, ay=-30)
            fig.add_vline(x=day, line_dash="dash", line_color="red", opacity=0.6)

    intervals = []
    curr_phase = data.iloc[0]['Phase']
    start = data.iloc[0]['Step']
    for i in range(1, len(data)):
        if data.iloc[i]['Phase'] != curr_phase:
            intervals.append((start, data.iloc[i-1]['Step'], curr_phase))
            curr_phase = data.iloc[i]['Phase']
            start = data.iloc[i]['Step']
    intervals.append((start, data.iloc[-1]['Step'], curr_phase))

    colors = px.colors.qualitative.Set2
    phase_color = {}
    for _, _, ph in intervals:
        if ph not in phase_color:
            phase_color[ph] = colors[len(phase_color) % len(colors)]

    for s, e, ph in intervals:
        fig.add_vrect(x0=s, x1=e, fillcolor=phase_color[ph], opacity=0.2,
                      layer="below", line_width=0,
                      annotation_text=ph, annotation_position="top left")

    if crop:
        for ph_name, ph_info in crop.phases.items():
            gdd_req = ph_info.gdd_required
            if gdd_req > 0 and gdd_req < data['GDD'].max():
                fig.add_hline(y=gdd_req, line_dash="dot", line_color="green", opacity=0.5,
                              annotation_text=f"{ph_name}: {gdd_req}°C·день")

    fig.update_layout(
        title=f"Накопление GDD и фенологические фазы (поле {field_id})",
        xaxis_title="День года",
        yaxis_title="Накопленная GDD (°C·день)",
        height=500
    )
    return fig

def plot_daily_stress_and_yield(agent_df: pd.DataFrame, weather_df: pd.DataFrame, crop: Crop, field_id: int,
                                field_capacity: float = 350, wilting_point: float = 150) -> go.Figure:
  
    if agent_df is None or agent_df.empty:
        return go.Figure()
    
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
    
    data = flat[flat['AgentID'] == field_id].copy()
    if data.empty:
        return go.Figure()
    
    required = ['GDD', 'Phase', 'SoilMoisture']
    if not all(col in data.columns for col in required):
        return go.Figure()
    
    data = data.sort_values('Step')
    data = data[data['Phase'].notna() & (data['Phase'] != '')]
    if data.empty:
        return go.Figure()
    
    weather_df['day'] = weather_df['day'].astype(int)
    data = data.merge(weather_df, left_on='Step', right_on='day', how='left')
    if data.empty or 'tmin' not in data.columns:
        return go.Figure()
    
    data['tmean'] = (data['tmin'] + data['tmax']) / 2
    
    days = data['Step'].values
    phases = data['Phase'].values
    gdd = data['GDD'].values
    tmean = data['tmean'].values
    precip = data['precip'].values
    par = data['par'].values
    soil_moist = data['SoilMoisture'].values
    
    stress_temp = np.zeros(len(data))
    stress_water = np.zeros(len(data))
    stress_par = np.zeros(len(data))
    overall_stress = np.zeros(len(data))
    
    phase_stress_sum = 0.0
    phase_days = 0
    cum_productivity = 1.0  
    current_phase = None
    phase_accum_stress = np.zeros(len(data))
    current_productivity = np.zeros(len(data))
    
    for idx, row in data.iterrows():
        phase_name = row['Phase']
        phase_info = None
        for ph_enum, ph_info in crop.phases.items():
            if ph_enum == phase_name or ph_info.phase_enum == phase_name:
                phase_info = ph_info
                break
        
        if phase_name != current_phase and current_phase is not None and phase_days > 0:
            avg_stress = phase_stress_sum / phase_days
            cum_productivity *= avg_stress
            phase_stress_sum = 0.0
            phase_days = 0
        current_phase = phase_name
        
        stress_t = temperature_stress(row['tmean'], crop=crop, phase_info=phase_info)
        
        sm = row['SoilMoisture']
        if sm >= field_capacity:
            stress_w = 1.0
        elif sm <= wilting_point:
            stress_w = 0.0
        else:
            stress_w = (sm - wilting_point) / (field_capacity - wilting_point)
        
        stress_p = par_stress_with_phase(row['par'], phase_info=phase_info, crop=crop)
        
        if phase_info and phase_info.weights:
            w = phase_info.weights
            total_w = sum(w.values())
            if total_w > 0:
                overall = (w.get('temp', 0)*stress_t +
                           w.get('water', 0)*stress_w +
                           w.get('par', 0)*stress_p) / total_w
            else:
                overall = 1.0
        else:
            overall = (stress_t + stress_w + stress_p) / 3.0
        
        stress_temp[idx] = stress_t
        stress_water[idx] = stress_w
        stress_par[idx] = stress_p
        overall_stress[idx] = overall
        
        phase_stress_sum += overall
        phase_days += 1
        phase_accum_stress[idx] = phase_stress_sum / phase_days
        current_productivity[idx] = cum_productivity * phase_accum_stress[idx]
    
    data['stress_temp'] = stress_temp
    data['stress_water'] = stress_water
    data['stress_par'] = stress_par
    data['overall_stress'] = overall_stress
    data['phase_accum_stress'] = phase_accum_stress
    data['productivity'] = current_productivity
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=days, y=stress_temp,
        mode='lines', name='Температурный',
        line=dict(color='red', width=2),
        hovertemplate='День %{x}<br>Темп. стресс: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=days, y=stress_water,
        mode='lines', name='Водный',
        line=dict(color='blue', width=2),
        hovertemplate='День %{x}<br>Водный стресс: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=days, y=stress_par,
        mode='lines', name='PAR',
        line=dict(color='green', width=2),
        hovertemplate='День %{x}<br>PAR стресс: %{y:.2f}<extra></extra>'
    ))
    
    customdata = np.stack((
        phases, gdd, tmean, precip, par, soil_moist,
        stress_temp, stress_water, stress_par,
        phase_accum_stress, current_productivity
    ), axis=-1)
    
    fig.add_trace(go.Scatter(
        x=days, y=overall_stress,
        mode='lines', name='Общий',
        line=dict(color='black', width=2.5, dash='dot'),
        hovertemplate='<b>День %{x}</b><br>' +
                      'Фаза: %{customdata[0]}<br>' +
                      'GDD: %{customdata[1]:.0f}°C·день<br>' +
                      'Температура: %{customdata[2]:.1f}°C<br>' +
                      'Осадки: %{customdata[3]:.1f} мм<br>' +
                      'PAR: %{customdata[4]:.1f} МДж/м²<br>' +
                      'Влажн. почвы: %{customdata[5]:.0f} мм<br>' +
                      '<b>Стрессы:</b><br>' +
                      'Температурный: %{customdata[6]:.2f}<br>' +
                      'Водный: %{customdata[7]:.2f}<br>' +
                      'PAR: %{customdata[8]:.2f}<br>' +
                      '<b>Общий стресс: %{y:.2f}</b><br>' +
                      'Средний стресс за фазу: %{customdata[9]:.2f}<br>' +
                      'Коэфф. продуктивности: %{customdata[10]:.2f}<br>' +
                      '<extra></extra>',
        customdata=customdata
    ))
    
    fig.add_hline(y=0.6, line_dash="dash", line_color="gray",
                  annotation_text="Порог стресса 0.6", annotation_position="bottom right")
    
    fig.update_layout(
        title=f"Ежедневные стрессы (поле {field_id}, культура {crop.name})",
        xaxis_title="День года",
        yaxis_title="Стресс (0..1)",
        height=500,
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

