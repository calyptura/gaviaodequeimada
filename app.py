import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
from io import StringIO
import datetime
import calendar
from datetime import datetime, timedelta
import os
import h3
import zipfile
import tempfile
from xml.etree import ElementTree as ET
import json

from streamlit import subheader

# Configuração da página
st.set_page_config(
    page_title="Painel interativo do monitoramento de fogo",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar API key do Mapbox
os.environ[
    "MAPBOX_API_KEY"] = "pk.eyJ1IjoiY2FseXB0dXJhIiwiYSI6ImNpdjV2MjhyNDAxaWMyb3MydHVvdTNhYXEifQ.zYAN0zIEFHZImB5xE_U3qg"


# Função para carregar dados do Google Sheets
@st.cache_data(ttl=3600)  # Cache por 1 hora
def carregar_dados_queimadas():
    """Carrega dados de queimadas do Google Sheets"""
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT4M8giX_sdB2b8y5S-GdlZ1A6PQPTIczB_x1hIX_By-v0K7WaxpJ8NsYFN8MVGqMgHZ9DmrilqeIra/pub?output=csv"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            csv_content = StringIO(response.content.decode('utf-8'))
            df = pd.read_csv(csv_content)

            # Processar as datas
            df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
            df['ano'] = df['acq_date'].dt.year
            df['mes'] = df['acq_date'].dt.month
            df['mes_nome'] = df['acq_date'].dt.month_name()
            df['ano_mes'] = df['acq_date'].dt.to_period('M')

            # Garantir que latitude e longitude são numéricas
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

            # Remover registros com coordenadas inválidas
            df = df.dropna(subset=['latitude', 'longitude', 'acq_date'])

            return df
        else:
            st.error(f"Erro ao carregar dados: Status {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


# Função para filtrar dados por período
def filtrar_por_periodo(df, periodo_selecionado, data_inicio=None, data_fim=None):
    """Filtra o DataFrame por período selecionado"""
    hoje = datetime.now()

    if periodo_selecionado == "Últimos 15 dias":
        data_limite = hoje - timedelta(days=15)
        return df[df['acq_date'] >= data_limite]
    elif periodo_selecionado == "Último mês":
        data_limite = hoje - timedelta(days=30)
        return df[df['acq_date'] >= data_limite]
    elif periodo_selecionado == "Série completa":
        return df
    elif periodo_selecionado == "Período personalizado":
        if data_inicio and data_fim:
            return df[(df['acq_date'] >= pd.to_datetime(data_inicio)) &
                      (df['acq_date'] <= pd.to_datetime(data_fim))]
        else:
            return df
    return df


# Função para verificar se um ponto está dentro do polígono (algoritmo ray casting)
def ponto_dentro_poligono(lat, lon, polygon_coords):
    """Verifica se um ponto está dentro de um polígono usando ray casting"""
    x, y = lon, lat
    n = len(polygon_coords)
    inside = False

    p1x, p1y = polygon_coords[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_coords[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


# Função para filtrar dados pelos limites municipais
def filtrar_dados_por_limites(df, limites_geojson):
    """Filtra DataFrame mantendo apenas pontos dentro dos limites municipais"""
    if limites_geojson is None or df.empty:
        return df

    # Obter coordenadas do polígono dos limites
    polygon_coords = limites_geojson['features'][0]['geometry']['coordinates'][0]

    # Verificar cada ponto
    mask = df.apply(
        lambda row: ponto_dentro_poligono(row['latitude'], row['longitude'], polygon_coords),
        axis=1
    )

    return df[mask]


# Função para calcular bounds do polígono KMZ
def calcular_bounds_kmz(limites_geojson):
    """Calcula os bounds (min/max lat/lon) do polígono KMZ"""
    if limites_geojson is None:
        return None

    try:
        coords = limites_geojson['features'][0]['geometry']['coordinates'][0]
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]

        return {
            'min_lon': min(lons),
            'max_lon': max(lons),
            'min_lat': min(lats),
            'max_lat': max(lats),
            'center_lon': (min(lons) + max(lons)) / 2,
            'center_lat': (min(lats) + max(lats)) / 2
        }
    except:
        return None


# Função para carregar limites municipais do KMZ
@st.cache_data(ttl=3600 * 24)  # Cache por 24 horas
def carregar_limites_cunha():
    """Carrega limites municipais de Cunha do Google Drive"""
    # URL do Google Drive convertida para download direto
    url_kmz = "https://drive.google.com/uc?id=1-BqKZ5DttphmyoWBy9TV4kqWaiPIbabl&export=download"

    try:
        # Download do arquivo KMZ
        response = requests.get(url_kmz, timeout=30)
        if response.status_code != 200:
            st.warning("Não foi possível carregar os limites municipais.")
            return None

        # Salvar temporariamente o KMZ
        with tempfile.NamedTemporaryFile(suffix='.kmz', delete=False) as temp_kmz:
            temp_kmz.write(response.content)
            temp_kmz_path = temp_kmz.name

        # Extrair KML do KMZ
        with zipfile.ZipFile(temp_kmz_path, 'r') as kmz:
            kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
            if not kml_files:
                st.warning("Arquivo KMZ não contém arquivos KML válidos.")
                return None

            # Ler o primeiro arquivo KML
            with kmz.open(kml_files[0]) as kml_file:
                kml_content = kml_file.read().decode('utf-8')

        # Limpar arquivo temporário
        os.unlink(temp_kmz_path)

        # Parse do KML
        root = ET.fromstring(kml_content)

        # Namespace do KML
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        # Encontrar coordenadas dos polígonos
        coordinates_elements = root.findall('.//kml:coordinates', ns)

        if not coordinates_elements:
            st.warning("Não foram encontradas coordenadas no arquivo KML.")
            return None

        # Converter coordenadas para formato GeoJSON
        features = []

        for coord_elem in coordinates_elements:
            coord_text = coord_elem.text.strip()
            if not coord_text:
                continue

            # Parse das coordenadas (formato: lon,lat,alt lon,lat,alt ...)
            points = []
            for point in coord_text.split():
                if point.strip():
                    coords = point.split(',')
                    if len(coords) >= 2:
                        lon, lat = float(coords[0]), float(coords[1])
                        points.append([lon, lat])

            if len(points) > 2:  # Polígono válido precisa de pelo menos 3 pontos
                # Fechar o polígono se necessário
                if points[0] != points[-1]:
                    points.append(points[0])

                feature = {
                    "type": "Feature",
                    "properties": {"name": "Cunha/SP"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [points]
                    }
                }
                features.append(feature)

        if features:
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }
            return geojson
        else:
            st.warning("Não foi possível extrair polígonos válidos do KML.")
            return None

    except Exception as e:
        st.warning(f"Erro ao carregar limites municipais: {e}")
        return None


# Função para calcular reincidência usando H3
def calcular_reincidencia_h3(df, tamanho_hex="1km"):
    """Calcula reincidência usando sistema H3 de hexágonos padronizados"""
    if df.empty:
        return pd.DataFrame()

    # Mapear tamanho para resolução H3
    resolucoes_h3 = {
        "1km": 7,  # ~1.22km de diâmetro
        "2km": 6,  # ~2.44km de diâmetro
        "5km": 5,  # ~4.25km de diâmetro
    }

    resolution = resolucoes_h3.get(tamanho_hex, 7)

    # Converter coordenadas para índices H3
    df_h3 = df.copy()
    df_h3['h3_index'] = df_h3.apply(
        lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], resolution),
        axis=1
    )

    # Agrupar por índice H3 e calcular estatísticas
    agrupamento = df_h3.groupby('h3_index').agg({
        'acq_date': ['count', 'min', 'max'],
        'satellite': lambda x: ', '.join(sorted(x.unique())[:3]),
        'brightness': 'mean',
        'confidence': 'first'
    }).reset_index()

    # Achatar colunas
    agrupamento.columns = ['h3_index', 'total_focos', 'data_min', 'data_max',
                           'satellites', 'brightness_media', 'confidence']

    # Calcular anos únicos separadamente para evitar problemas de serialização
    anos_por_hex = df_h3.groupby('h3_index')['acq_date'].apply(
        lambda x: sorted(list(x.dt.year.unique()))
    ).reset_index()
    anos_por_hex.columns = ['h3_index', 'anos_list']

    # Merge com dados principais
    agrupamento = agrupamento.merge(anos_por_hex, on='h3_index')

    # Calcular informações de reincidência como strings
    agrupamento['anos_str'] = agrupamento['anos_list'].apply(lambda x: ', '.join(map(str, x)))
    agrupamento['num_anos'] = agrupamento['anos_list'].apply(len)
    agrupamento['reincidencia_info'] = agrupamento.apply(
        lambda row: f"{row['total_focos']} foco(s) em {row['num_anos']} ano(s) ({row['anos_str']})",
        axis=1
    )

    # Calcular período como string
    agrupamento['periodo'] = agrupamento['data_min'].dt.strftime('%m/%Y') + ' a ' + agrupamento['data_max'].dt.strftime(
        '%m/%Y')

    # Obter coordenadas dos hexágonos H3
    coordenadas = []
    polygons = []
    areas_info = []

    for h3_index in agrupamento['h3_index']:
        # Centroide
        lat, lon = h3.cell_to_latlng(h3_index)
        coordenadas.append((lat, lon))

        # Polígono - converter para lista de listas simples
        boundary = h3.cell_to_boundary(h3_index)
        polygon = [[float(coord[1]), float(coord[0])] for coord in boundary]  # [lon, lat]
        polygons.append(polygon)

        # Área
        area_km2 = h3.cell_area(h3_index, unit='km^2')
        areas_info.append(f"{tamanho_hex} (~{area_km2:.1f}km²)")

    # Adicionar coordenadas
    agrupamento['latitude'] = [coord[0] for coord in coordenadas]
    agrupamento['longitude'] = [coord[1] for coord in coordenadas]
    agrupamento['polygon'] = polygons
    agrupamento['area_info'] = areas_info

    # Converter tipos para serialização JSON
    agrupamento['brightness_media'] = agrupamento['brightness_media'].round(1).astype(float)
    agrupamento['total_focos'] = agrupamento['total_focos'].astype(int)
    agrupamento['h3_index'] = agrupamento['h3_index'].astype(str)

    # Remover colunas que podem causar problemas de serialização
    colunas_finais = ['h3_index', 'latitude', 'longitude', 'polygon', 'total_focos',
                      'reincidencia_info', 'periodo', 'satellites', 'brightness_media',
                      'area_info', 'num_anos']

    return agrupamento[colunas_finais]


# Função para gerar mapa de queimadas
def gerar_mapa_queimadas(df, tipo_mapa, estilo_mapa, transparencia=0.8, tamanho_hex="1km", limites_geojson=None):
    """Gera mapa de queimadas usando PyDeck"""
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None

    # Filtrar coordenadas válidas
    df_mapa = df.dropna(subset=['latitude', 'longitude']).copy()

    if len(df_mapa) == 0:
        return None

    # Calcular centro do mapa baseado nos limites KMZ ou nos dados
    if limites_geojson:
        bounds = calcular_bounds_kmz(limites_geojson)
        if bounds:
            center_lat = bounds['center_lat']
            center_lon = bounds['center_lon']
            # Calcular zoom adequado baseado na área do polígono
            lat_range = bounds['max_lat'] - bounds['min_lat']
            lon_range = bounds['max_lon'] - bounds['min_lon']
            max_range = max(lat_range, lon_range)

            # Ajustar zoom baseado no range (valores empíricos)
            if max_range > 1.0:
                zoom = 8
            elif max_range > 0.5:
                zoom = 9
            elif max_range > 0.2:
                zoom = 10
            else:
                zoom = 11
        else:
            center_lat = df_mapa['latitude'].mean()
            center_lon = df_mapa['longitude'].mean()
            zoom = 11
    else:
        center_lat = df_mapa['latitude'].mean()
        center_lon = df_mapa['longitude'].mean()
        zoom = 11

    # Definir estilos de mapa
    estilos_mapa = {
        "Satélite": "mapbox://styles/mapbox/satellite-v9",
        "Claro": "mapbox://styles/mapbox/light-v11",
        "Escuro": "mapbox://styles/mapbox/dark-v11"
    }

    estilo_url = estilos_mapa.get(estilo_mapa, "mapbox://styles/mapbox/satellite-v9")

    # Configurar visualização inicial
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
        bearing=0
    )

    # Criar camadas baseadas no tipo de mapa
    layers = []

    if tipo_mapa == "Mapa de pontos":
        # Para pontos, usar dados originais (registro específico)
        # Formatear a data para exibição
        df_mapa['data_formatada'] = df_mapa['acq_date'].dt.strftime('%d/%m/%Y')
        df_mapa['hora_formatada'] = df_mapa['acq_time'].astype(str).str.zfill(4)
        df_mapa['hora_formatada'] = df_mapa['hora_formatada'].str[:2] + ':' + df_mapa['hora_formatada'].str[2:]

        layers.append(pdk.Layer(
            'ScatterplotLayer',
            data=df_mapa,
            get_position=['longitude', 'latitude'],
            get_color=[255, 100, 0, 200],  # Cor laranja para fogo
            get_radius=50,
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_scale=6,
            radius_min_pixels=3,
            radius_max_pixels=15,
            line_width_min_pixels=1,
        ))

        # Tooltip específico para pontos (registro individual)
        tooltip = {
            "html": """
            <div style="padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px; max-width: 250px;">
              <b>📅 Data:</b> {data_formatada}<br/>
              <b>🕐 Hora:</b> {hora_formatada}<br/>
              <b>🛰️ Satélite:</b> {satellite}<br/>
              <b>🔧 Instrumento:</b> {instrument}<br/>
              <b>🔆 Brilho:</b> {brightness}<br/>
              <b>✅ Confiança:</b> {confidence}<br/>
              <b>🌡️ FRP:</b> {frp}
            </div>
            """,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }

    elif tipo_mapa == "Mapa de calor":
        layers.append(pdk.Layer(
            'HeatmapLayer',
            data=df_mapa,
            get_position=['longitude', 'latitude'],
            opacity=0.8,
            get_weight=1,
            radiusPixels=60,
            color_range=[
                [255, 255, 178],  # Amarelo claro
                [254, 204, 92],  # Amarelo
                [253, 141, 60],  # Laranja
                [240, 59, 32],  # Vermelho
                [189, 0, 38]  # Vermelho escuro
            ]
        ))

        # Tooltip básico para mapa de calor
        tooltip = {
            "html": """
            <div style="padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px;">
              <b>🔥 Visualização de densidade de focos</b><br/>
              <b>💡 Dica:</b> Cores mais quentes = maior concentração
            </div>
            """,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }

    elif tipo_mapa == "Mapa de hexágono":
        # Usar sistema H3 para hexágonos padronizados com reincidência
        df_h3_reincidencia = calcular_reincidencia_h3(df_mapa, tamanho_hex)

        if df_h3_reincidencia.empty:
            st.warning("Não foi possível calcular hexágonos H3 para os dados filtrados.")
            return None

        # Usar PolygonLayer para desenhar hexágonos H3 reais
        layers.append(pdk.Layer(
            'PolygonLayer',
            data=df_h3_reincidencia,
            get_polygon='polygon',
            get_fill_color=[255, 100, 0, int(255 * transparencia)],  # Cor com transparência
            get_line_color=[255, 255, 255, 200],
            line_width=2,
            pickable=True,
            auto_highlight=True,
            get_elevation=0,
        ))

        # Criar escala de cores baseada na contagem de focos
        max_focos = df_h3_reincidencia['total_focos'].max() if len(df_h3_reincidencia) > 0 else 1

        # Função para calcular cor baseada no número de focos
        def get_color_by_count(count, max_count):
            # Normalizar entre 0 e 1
            normalized = min(count / max_count, 1.0)

            if normalized <= 0.2:
                return [255, 255, 178, int(255 * transparencia)]  # Amarelo claro
            elif normalized <= 0.4:
                return [254, 204, 92, int(255 * transparencia)]  # Amarelo
            elif normalized <= 0.6:
                return [253, 141, 60, int(255 * transparencia)]  # Laranja
            elif normalized <= 0.8:
                return [240, 59, 32, int(255 * transparencia)]  # Vermelho
            else:
                return [189, 0, 38, int(255 * transparencia)]  # Vermelho escuro

        # Aplicar cores baseadas na contagem
        df_h3_reincidencia['fill_color'] = df_h3_reincidencia['total_focos'].apply(
            lambda x: get_color_by_count(x, max_focos)
        )

        # Atualizar layer com cores dinâmicas
        layers[-1] = pdk.Layer(
            'PolygonLayer',
            data=df_h3_reincidencia,
            get_polygon='polygon',
            get_fill_color='fill_color',
            get_line_color=[255, 255, 255, 200],
            line_width=1,
            pickable=True,
            auto_highlight=True,
            get_elevation=0,
        )

        # Tooltip para hexágonos H3 com informações completas de reincidência
        tooltip = {
            "html": """
            <div style="padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px; max-width: 300px;">
              <b>🔥 Focos neste hexágono H3:</b> {total_focos}<br/>
              <b>🔄 Reincidência:</b> {reincidencia_info}<br/>
              <b>📅 Período:</b> {periodo}<br/>
              <b>🛰️ Satélites:</b> {satellites}<br/>
              <b>🔆 Brilho médio:</b> {brightness_media}<br/>
              <b>📏 Área:</b> {area_info}<br/>
              <b>🆔 ID H3:</b> {h3_index}
            </div>
            """,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }

    # Adicionar camada de limites municipais (sempre ativado)
    if limites_geojson:
        # Configurações fixas para os limites
        cor_rgb = [199, 193, 193]  # #C7C1C1
        espessura_linha = 2

        # Extrair coordenadas do polígono
        polygon_coords = limites_geojson['features'][0]['geometry']['coordinates'][0]

        # Preparar dados para PathLayer
        path_data = [{
            'path': polygon_coords,
            'color': cor_rgb + [255],
            'width': espessura_linha
        }]

        layers.append(pdk.Layer(
            'PathLayer',
            data=path_data,
            get_path='path',
            get_color='color',
            get_width='width',
            width_scale=1,
            width_min_pixels=espessura_linha,
            pickable=False,
            auto_highlight=False
        ))

    # Criar deck
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=estilo_url,
        map_provider="mapbox",
        tooltip=tooltip
    )

    return r


# Função para gerar gráfico temporal
def gerar_grafico_temporal(df):
    """Gera gráfico de barras com série temporal mensal"""
    if df.empty or 'acq_date' not in df.columns:
        return None

    # Calcular data limite (últimos 12 meses)
    hoje = datetime.now()
    data_limite = hoje - timedelta(days=365)

    # Filtrar últimos 12 meses
    df_temporal = df[df['acq_date'] >= data_limite].copy()

    if df_temporal.empty:
        return None

    # Agrupar por mês
    contagem_mensal = df_temporal.groupby(df_temporal['acq_date'].dt.to_period('M')).size().reset_index()
    contagem_mensal.columns = ['Período', 'Ocorrências']

    # Converter período para string
    contagem_mensal['Mês'] = contagem_mensal['Período'].dt.strftime('%b/%Y')

    # Criar gráfico de barras
    fig = px.bar(
        contagem_mensal,
        x='Mês',
        y='Ocorrências',
        title='Série Temporal de Queimadas - Últimos 12 Meses',
        color='Ocorrências',
        color_continuous_scale='Reds'
    )

    # Customizar layout
    fig.update_layout(
        xaxis_title='Mês/Ano',
        yaxis_title='Número de Ocorrências',
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(tickangle=-45)
    )

    return fig


# Interface principal
def main():
    # Título principal
    st.title("Painel interativo de monitoramento de fogo")
    st.subheader("Cunha - SP")
    st.markdown("---")

    # Carregar dados e limites municipais
    with st.spinner("Carregando dados de queimadas e limites municipais..."):
        df = carregar_dados_queimadas()
        limites_geojson = carregar_limites_cunha()

    if df.empty:
        st.error("Não foi possível carregar os dados. Verifique sua conexão de internet.")
        return

    # Sidebar - Filtros (com logo no topo)
    with st.sidebar:
        # Logo do projeto no topo
        st.title("gavião-de-queimada: vendo o fogo do alto")
        st.image("https://i.imgur.com/JgQKCU7.png", width=200)
        st.markdown("---")

        # Filtro por período (padrão: Série completa)
        st.markdown("### Período de Análise")
        periodo_opcoes = ["Série completa", "Último mês", "Últimos 15 dias", "Período personalizado"]
        periodo_selecionado = st.selectbox("Selecione o período:", periodo_opcoes, index=0)

        # Campos de data personalizada
        data_inicio = None
        data_fim = None
        if periodo_selecionado == "Período personalizado":
            col1, col2 = st.columns(2)
            with col1:
                data_inicio = st.date_input("Data inicial:")
            with col2:
                data_fim = st.date_input("Data final:")

        # Informações dos dados
        st.markdown("---")
        # Aplicar filtros
        df_filtrado = df.copy()

        # Filtrar por período
        df_filtrado = filtrar_por_periodo(df_filtrado, periodo_selecionado, data_inicio, data_fim)

        # Filtrar por limites municipais (sempre ativado)
        if limites_geojson:
            df_filtrado = filtrar_dados_por_limites(df_filtrado, limites_geojson)

        # Exibir estatísticas
        total_ocorrencias = len(df_filtrado)
        if not df_filtrado.empty:
            data_mais_antiga = df_filtrado['acq_date'].min().strftime('%d/%m/%Y')
            data_mais_recente = df_filtrado['acq_date'].max().strftime('%d/%m/%Y')

            st.metric("Total de Ocorrências", total_ocorrencias)
            st.text(f"Período: {data_mais_antiga} a {data_mais_recente}")

            st.markdown("---")

            # Informação sobre satélites utilizados

            # Calcular período dos dados dinamicamente
            if not df.empty and 'acq_date' in df.columns:
                data_mais_antiga = df['acq_date'].min().strftime('%d/%m/%Y')
                satelites_disponiveis = sorted(df['satellite'].unique()) if 'satellite' in df.columns else []
                satelites_str = " e ".join(satelites_disponiveis) if satelites_disponiveis else "N/A"

                st.info(
                    f"Informações com base nos satélites **{satelites_str}**, com dados atualizados diariamente desde **{data_mais_antiga}**. "
                    f"Origem dos dados: NASA FIRMS")
            else:
                st.info("São utilizados satélites com dados atualizados diariamente.")

        else:
            st.warning("Nenhum registro encontrado com os filtros aplicados.")

    # Layout principal - Uma coluna (mapa em largura total)

    # Seção do mapa

    # Controles do mapa
    map_col1, map_col2, map_col3 = st.columns(3)

    with map_col1:
        tipo_mapa = st.selectbox(
            "Tipo de Mapa:",
            ["Mapa de hexágono", "Mapa de calor", "Mapa de pontos"],
            index=0  # Define hexágono como padrão
        )

    with map_col2:
        estilo_mapa = st.selectbox(
            "Estilo do Mapa:",
            ["Satélite", "Claro", "Escuro"]
        )

    with map_col3:
        # Controle de transparência apenas para hexágonos
        if tipo_mapa == "Mapa de hexágono":
            transparencia = st.slider(
                "Transparência:",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="Ajuste a transparência dos hexágonos"
            )

            # Controle de tamanho dos hexágonos (removido 10km)
            tamanho_hex = st.selectbox(
                "Tamanho dos hexágonos:",
                options=["1km", "2km", "5km"],
                index=0,  # Padrão: 1km
                help="Selecione o tamanho dos hexágonos"
            )
        else:
            transparencia = 0.8  # Valor padrão para outros mapas
            tamanho_hex = "1km"  # Valor padrão

    # Gerar e exibir mapa (largura total)
    if not df_filtrado.empty:
        mapa = gerar_mapa_queimadas(df_filtrado, tipo_mapa, estilo_mapa, transparencia, tamanho_hex, limites_geojson)
        if mapa:
            st.pydeck_chart(mapa)
        else:
            st.warning("Não foi possível gerar o mapa com os dados filtrados.")
    else:
        st.warning("Nenhum dado disponível para exibir no mapa.")

    # Gráfico temporal com Top 5 Dias ao lado
    st.markdown("---")
    st.subheader("Série Temporal de Queimadas")

    if not df.empty:  # Usar dados completos para o gráfico temporal
        # Criar duas colunas: gráfico na esquerda, top 5 dias na direita
        chart_col, stats_col = st.columns([3, 1])

        with chart_col:
            fig_temporal = gerar_grafico_temporal(df)
            if fig_temporal:
                st.plotly_chart(fig_temporal, use_container_width=True)
            else:
                st.warning("Não foi possível gerar o gráfico temporal.")

        with stats_col:
            # Top 5 dias com mais ocorrências (usando dados filtrados)
            if not df_filtrado.empty and 'acq_date' in df_filtrado.columns:
                st.markdown("**Top 5 Dias:**")
                top_dias = df_filtrado['acq_date'].dt.date.value_counts().head(5).reset_index()
                top_dias.columns = ['Data', 'Ocorrências']

                for _, row in top_dias.iterrows():
                    st.text(f"📅 {row['Data']}: {row['Ocorrências']}")
            else:
                st.markdown("**Top 5 Dias:**")
                st.text("Nenhum dado disponível")
    else:
        st.warning("Nenhum dado disponível para o gráfico temporal.")

    # Rodapé
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; font-size: 0.8rem; color: gray;">
            Painel interativo de monitoramento de fogo - Cunha - SP | Dados: NASA FIRMS | Desenvolvido por Luciano Lima<br/>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
