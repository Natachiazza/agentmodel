import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import re


def get_agent_data(agent_df: pd.DataFrame, variable_name: str):
    if agent_df is None or agent_df.empty:
        return None
    if agent_df.index.name == 'Step' or 'Step' in agent_df.index.names:
        agent_df = agent_df.reset_index()
    if 'Step' not in agent_df.columns:
        agent_df = agent_df.reset_index().rename(columns={'index': 'Step'})
    # MultiIndex колонок (AgentID, Variable)
    if isinstance(agent_df.columns, pd.MultiIndex):
        mask = [col[1] == variable_name for col in agent_df.columns]
        if any(mask):
            subset = agent_df.loc[:, mask]
            subset.columns = [col[0] for col in subset.columns]
            return subset.melt(id_vars='Step', var_name='AgentID', value_name=variable_name)
        
    if variable_name in agent_df.columns:
        if 'AgentID' in agent_df.columns:
            return agent_df[['Step', 'AgentID', variable_name]]
        else:
            value_cols = [c for c in agent_df.columns if c != 'Step']
            if value_cols:
                return agent_df.melt(id_vars='Step', var_name='AgentID', value_name=variable_name)
    return None

def plot_soil_moisture(agent_df: pd.DataFrame, field_ids: List[int] = None):
  
    data = get_agent_data(agent_df, 'SoilMoisture')
    if data is None or data.empty:
        return None
    data = data.dropna(subset=['SoilMoisture'])
    if field_ids:
        data = data[data['AgentID'].isin(field_ids)]
    fig = px.line(data, x='Step', y='SoilMoisture', color='AgentID',
                  title="Влажность почвы по дням")
    fig.update_layout(xaxis_title="День", yaxis_title="Влажность, мм")
    return fig

def plot_machine_utilization(agent_df: pd.DataFrame):
    status_data = get_agent_data(agent_df, 'MachineStatus')
    type_data = get_agent_data(agent_df, 'MachineType')
    if status_data is None or type_data is None:
        return None
    merged = status_data.merge(type_data, on=['Step','AgentID'], suffixes=('_status','_type'))
    merged = merged.dropna(subset=['MachineType'])
    if merged.empty:
        return None
    merged['IsBusy'] = (merged['MachineStatus'] == 'busy').astype(int)
    busy_by_step_type = merged.groupby(['Step', 'MachineType'])['IsBusy'].mean().reset_index()
    fig = px.line(busy_by_step_type, x='Step', y='IsBusy', color='MachineType',
                  title="Доля занятой техники по типам")
    fig.update_layout(yaxis_title="Доля занятости", xaxis_title="День")
    return fig


def plot_gantt_by_operations(operations_df: pd.DataFrame, field_colors: Dict[int, str] = None) -> go.Figure:
    if operations_df.empty:
        return go.Figure()
    
    df = operations_df.copy()
    df = df.sort_values('start_day')
    df['duration'] = df['end_day'] - df['start_day'] + 1
    df['duration'] = df['duration'].clip(lower=1)
    
    df['y_label'] = df['operation_name'] + ' (поле ' + df['field_id'].astype(str) + ')'
    
    if field_colors is None:
        unique_fields = df['field_id'].unique()
        palette = px.colors.qualitative.Plotly
        field_colors = {fid: palette[i % len(palette)] for i, fid in enumerate(unique_fields)}
    
    fig = go.Figure()
    
    for _, row in df.iterrows():
        hover_text = (
            f"<b>{row['operation_name']}</b><br>"
            f"Поле: {row['field_id']}<br>"
            f"Дни: {row['start_day']}–{row['end_day']}<br>"
            f"Требуется машин: {row.get('required_machines', 1):.1f}<br>"
            f"Трудозатраты: {row.get('labor_hours', 0):.0f} чел·ч<br>"
            f"Рабочих: {row.get('worker_count', 0)}<br>"
            f"Затраты: {row.get('cost', 0):,.0f} руб<br>"
            f"Статус: {row.get('status', 'planned')}"
        )
        fig.add_trace(go.Bar(
            x=[row['duration']],
            y=[row['y_label']],
            base=row['start_day'],
            orientation='h',
            marker_color=field_colors[row['field_id']],
            opacity=0.8,
            text=row['operation_name'],
            hovertext=hover_text,
            hoverinfo='text',
            name=f"Поле {row['field_id']}"
        ))
    
    fig.update_yaxes(
        categoryorder='array',
        categoryarray=df['y_label'].tolist()[::-1]   # обратный порядок
    )
    
    fig.update_layout(
        title="Диаграмма Ганта агротехнических операций",
        xaxis_title="День года",
        yaxis_title="Операция",
        barmode='stack',
        height=30 * len(df) + 100,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

def plot_yield_by_scenario(scenario_results: Dict[str, Dict[str, float]]) -> go.Figure:
    
    #scenario_results: {сценарий: {культура: урожайность}}
    if not scenario_results:
        return go.Figure()
    data = []
    for scen, yields in scenario_results.items():
        for crop, yld in yields.items():
            data.append({'Сценарий': scen, 'Культура': crop, 'Урожайность (т/га)': yld})
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure()
    fig = px.bar(df, x='Культура', y='Урожайность (т/га)', color='Сценарий', barmode='group',
                 title="Урожайность по сценариям")
    return fig


def plot_weather(weather_data, title="Погодные условия"):
    if weather_data is None:
        return None
    
    if isinstance(weather_data, list):
        if len(weather_data) == 0:
            return None
        df = pd.DataFrame(weather_data)
    else:
        df = weather_data.copy()
    
    if df.empty:
        return None
    
    if 'day' not in df.columns:
        return None
    
    df = df.sort_values('day')
    fig = go.Figure()
    
    if 'tmin' in df.columns and 'tmax' in df.columns:
        df['tmean'] = (df['tmin'] + df['tmax']) / 2
        fig.add_trace(go.Scatter(
            x=df['day'], y=df['tmean'],
            mode='lines', name='Средняя темп., °C',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['day'], y=df['tmin'],
            mode='lines', name='Мин. темп., °C',
            line=dict(color='orange', width=1, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=df['day'], y=df['tmax'],
            mode='lines', name='Макс. темп., °C',
            line=dict(color='darkred', width=1, dash='dot')
        ))
    elif 'temperature' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['day'], y=df['temperature'],
            mode='lines', name='Температура, °C'
        ))
    
    if 'precip' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['day'], y=df['precip'],
            mode='lines', name='Осадки, мм',
            yaxis='y2', line=dict(color='blue', width=1)
        ))
    
    if 'par' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['day'], y=df['par'],
            mode='lines', name='PAR, МДж/м²',
            yaxis='y3', line=dict(color='green', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="День года",
        yaxis_title="Температура, °C",
        yaxis2=dict(
            title="Осадки, мм",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        yaxis3=dict(
            title="PAR, МДж/м²",
            overlaying='y',
            side='right',
            anchor='free',
            position=0.85,
            showgrid=False
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def display_detailed_adaptation_table(events_df: pd.DataFrame, operations_dict: Dict[int, any],
                                      tech_cards_df: pd.DataFrame) -> pd.DataFrame:
    
    if events_df is None or events_df.empty:
        return pd.DataFrame()
    adapt = events_df[events_df['event_type'] == "Адаптация"].copy()
    if adapt.empty:
        return pd.DataFrame()
    
    def parse_desc(desc):
        resource = None
        if 'лимит воды' in desc or 'вода' in desc:
            resource = 'вода'
        elif 'бюджет' in desc:
            resource = 'бюджет'
        elif 'техника' in desc:
            resource = 'техника'
        elif 'рабочий' in desc:
            resource = 'рабочий'
        op_match = re.search(r'операция (.+?),', desc)
        operation = op_match.group(1) if op_match else None
        field_match = re.search(r'поле (\d+)', desc)
        field = int(field_match.group(1)) if field_match else None
        req_match = re.search(r'требовалось (\d+(?:\.\d+)?)', desc)
        required = float(req_match.group(1)) if req_match else None
        avail_match = re.search(r'доступно (\d+(?:\.\d+)?)', desc)
        available = float(avail_match.group(1)) if avail_match else None
        # Увеличение
        inc_match = re.search(r'на (\d+(?:\.\d+)?)', desc)
        increase = float(inc_match.group(1)) if inc_match else None
        # Тип техники
        machine_match = re.search(r'техника (\w+)', desc)
        machine_type = machine_match.group(1) if machine_match else None
        return pd.Series({'resource': resource, 'operation': operation, 'field': field,
                          'required': required, 'available': available,
                          'increase': increase, 'machine_type': machine_type})
    
    details = adapt['description'].apply(parse_desc)
    result = pd.concat([adapt[['day']], details], axis=1)
    return result

def plot_rejection_reasons(event_df: pd.DataFrame) -> go.Figure:
    if event_df is None or event_df.empty:
        return go.Figure()
    df = event_df[event_df['event_type'] == 'Отказ'].copy()
    if df.empty:
        return go.Figure()
    df['reason'] = df['description'].str.extract(r'причина=(\w+)')[0]
    pivot = df.groupby(['day', 'reason']).size().unstack(fill_value=0)
    fig = px.bar(pivot, title="Причины отказов по дням", barmode='stack')
    return fig

def plot_downtime_events(event_df: pd.DataFrame) -> go.Figure:
    if event_df is None or event_df.empty:
        return go.Figure()
    df = event_df[event_df['event_type'] == 'Отказ'].copy()
    df = df[df['description'].str.contains('SOIL_MOISTURE', case=False, na=False)]
    if df.empty:
        return go.Figure()
    fig = px.scatter(df, x='day', y='agent_id', color='description',
                     title="Простои операций (переувлажнение)")
    return fig

