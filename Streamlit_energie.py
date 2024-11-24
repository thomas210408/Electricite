import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import geopandas as gpd
import matplotlib.colors as mcolors
from shapely import wkt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
import base64
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stat
from scipy.stats import pearsonr
from sklearn.metrics  import  r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from prophet.plot import plot_cross_validation_metric




df = pd.read_parquet('df.parquet')
df_ml=pd.read_csv('df_ml.csv', sep=',')
df_ml['date_heure_simplifiée'] = pd.to_datetime(df_ml['date_heure_simplifiée'])

df_sarima_idf=pd.read_csv('sarima_data_idf.csv', sep=',')
df_sarima_bfc2=pd.read_csv('sarima_data_bfc.csv', sep=',')

template=pd.read_csv('Template.csv', sep=',')




os.environ["STREAMLIT_CONFIG_FILE"] = "/Users/fran/Downloads/Streamlit Energie/config.toml"







st.title("Analyse et Prévision de la consommation Electrique Future en France : Modélisation et Tendances")


st.sidebar.title("Sommaire")
pages=["**Présentation**","**Exploration**", "**DataVizualization**", "**Modélisation**","**Perspective**"]
page=st.sidebar.radio("Aller vers", pages)





st.sidebar.markdown (':people_holding_hands:***:blue[Auteurs]***')
st.sidebar.write('*Legardinier François*')
st.sidebar.write('*Gendronneau Thomas*')






if page== pages[0] :
  st.write ("### Présentation")
  st.image("Energie_1.jpg")
  st.write ("""Depuis fin 2022, le gouvernement français alerte sur la possibilité de coupures d'électricité durant l'hiver. Ces interruptions, bien que limitées à des zones géographiques restreintes et n'excédant pas deux heures, soulèvent des questions importantes sur la résilience de notre réseau électrique. Ces coupures localisées, dues principalement à des pics de consommation, montrent que la France dispose encore de marges de manoeuvre, offrant un aperçu rassurant de notre capacité à gérer la demande en électricité. Cependant, cette situation nous pousse également à réfléchir sur notre aptitude à affronter des crises électriques futures potentiellement plus graves, exacerbées par nos habitudes de consommation et nos besoins croissants.

   Face à la variabilité de la demande et à l'intermittence des sources renouvelables, la France se trouve à un carrefour. Le pays s'appuie fortement sur l'énergie nucléaire pour son approvisionnement stable en électricité, tout en intégrant progressivement des énergies renouvelables pour diversifier son mix énergétique et réduire son empreinte carbone. Cette transition est essentielle non seulement pour répondre aux objectifs environnementaux mais aussi pour assurer une sécurité énergétique durable.

   Nous traversons actuellement une crise électrique attribuable à trois facteurs principaux :
   1. Augmentation du prix du gaz.
   2. Le changement climatique (notamment la sécheresse).
   3. Retard dans l'entretien des réacteurs nucléaires.
    
   Ces éléments ont catalysé une réflexion urgente sur la nécessité de réduire la dépendance aux énergies fossiles et de diminuer la dépendance énergétique vis-à-vis de la Russie. Le Sénat traite actuellement ce sujet au travers d'une commission d'enquête sur l'électricité, en réponse aux engagements de décarbonation pris lors de la COP22, visant une réduction de 35% des émissions de gaz à effet de serre d'ici 2030. La solution envisagée repose largement sur l'augmentation de l'électrification, bien que l'électricité ne soit pas stockable sur de longues périodes (stockage journalier possible).

   A travers ce projet nous allons essayer de prédire la consommation électrique future.
    """)
  

  st.image("Energie_17.png") 



if page == pages[1] : 
    
  st.image("Energie_8.png")   
  st.write("### Exploration")
 
  st.write("Note jeux de Donnée principale est : Données éCO2mix régionales consolidées et définitives (janvier 2013 à janvier 2023:")
  st.write ("""Le jeu de données Eco2Mix contient des informations complètes et détaillées sur la consommation électrique et la production pour différentes filières, incluant le nucléaire, l'éolien, le solaire et le thermique.Les données sont mises à jour toutes les demi-heures, offrant ainsi une granularité fine qui permet d'analyser les variations de consommation et de production en temps quasi-réel. Les données sont regroupées au niveau régional, ce qui permet une analyse précise des variations géographiques de la consommation et de la production électrique.""")

  st.write("**Affichage des 5 premières lignes :**")  

  st.dataframe(df.head())

  st.write ("Le nombre de ligne du jeu de donnée est :", df.shape[0])
  st.write ("Le nombre de colonne du jeu de donnée est :", df.shape[1])

  colonne= template['Nom de la colonne'].unique()

  st.write ("**Dictionnaire de donnée :**")
  selection_colonne=st.selectbox("Choissisez une colonne:", colonne)

  filtre_template=template[template ['Nom de la colonne']== selection_colonne]

  st.dataframe(filtre_template['Description'])

  st.write ('**Quelques statistiques**')
  
  st.dataframe(df['consommation'].describe())

  
  st.write ('**NA**')
  if st.checkbox("Afficher les NA") :
   st.dataframe(df.isna().sum())
  
  st.write ("""Le jeu de données Eco2Mix est d'excellente qualité, sans doublons et avec très peu de données manquantes ce qui nous permet d'avoir une bonne qualité dans nos analyses.Une compréhension approfondie du domaine a permis de repérer et de corriger des anomalies telles que les fausses NaN dans la production nucléaire, qui correspondaient en réalité à des régions dépourvues de centrales nucléaires. Ces NaN ont été remplacés par des valeurs nulles pour maintenir l'intégrité des données.""")
  
  st.write ("**Autres sources de données**")
  st.write ("""Nous avons récupéré d'autres jeux de données :
      ● Température (source : Température quotidienne régionale (depuis janvier 2016) - data.gouv.fr) : Il s'agit des températures moyennes, minimum et maximum par région et par jours.
     ● Secteurs d'activités : Consommation et thermosensibilité électriques annuelles à la maille région (source : https://data.enedis.fr/explore/dataset/consommation-electrique-par-secteur-dactivite-region/table/?sort=annee) : Il s'agit des consommations annuelles par secteur d'activité par région avec un détail sur le résidentiel (Années des logements, part thermosensibilité…).
     ● Hydraulique
     ● Gdf : Géographie des régions avec leurs coordonnées ainsi que leur superficie""" )

  st.write("""Les données étant très propre, nous n'avons pas eu besoin de nettoyer les données. Le jeu de données température fonctionne aussi de façon journalière tandis que les données sur le secteur d'activité sont des données annualisées.""")



if page == pages[2] : 
    
  st.image("Energie_11.jpg")  
  st.write("### DataVizualization")
  
  
  st.write("""
Dans le cadre de notre projet et avant d'entamer le processus de modélisation, il est essentiel de bien comprendre la répartition de la production énergétique entre les différentes sources ainsi que son évolution au cours des dernières années. Cette analyse préliminaire est fondamentale pour saisir les dynamiques actuelles du mix énergétique français et identifier les tendances qui pourraient influencer la consommation et la résilience du réseau dans les années à venir.

L'objectif de cette section est donc de fournir une visualisation claire et instructive de la répartition et des tendances des sources de production d'énergie, notamment pour anticiper et gérer les futures crises énergétiques. Nous examinons les données de production par source, en mettant en lumière le rôle de chaque type d'énergie (nucléaire, renouvelables, fossiles) dans le mix énergétique français et en identifiant les évolutions notables.
""")

  st.write("""
Les graphiques ci-dessous présentent la répartition de la production d'énergie en France de 2018 à 2022. Ces données permettent de visualiser la contribution relative de chaque source – nucléaire, hydraulique, thermique, éolien, solaire, et bioénergies – et d'observer leur évolution annuelle.
""")

  
  

  production_columns = ['thermique', 'nucleaire', 'eolien', 'solaire', 'hydraulique', 'bioenergies']
  for energy in production_columns:
      df[energy] = pd.to_numeric(df[energy], errors='coerce')


  df['date'] = pd.to_datetime(df['date'], errors='coerce')


  years = [2018, 2019, 2020, 2021, 2022]
  selected_year = st.selectbox("Sélectionnez l'année pour la répartition de la production d'énergie :", years)


  df_selected_year = df[df['date'].dt.year == selected_year]


  production_totale_par_type = df_selected_year[production_columns].sum()


  production_totale_par_type = production_totale_par_type.dropna().sort_values(ascending=False)


  colors = ['green', 'orange', 'lightgreen', 'red', 'skyblue', 'blue']


  st.write(f"### Répartition de la Production d'Énergie française par Source en {selected_year}")
  plt.figure(figsize=(8, 6))
  production_totale_par_type.plot(kind='pie', colors=colors, autopct='%1.1f%%', startangle=140)
  plt.ylabel('')
  plt.title(f"Répartition de la Production d'Énergie française par Source en {selected_year}")
  st.pyplot(plt)
  
  
  
  st.write("""
Le mix énergétique français est dominé par le nucléaire, offrant une production stable et à faible empreinte carbone. Les énergies renouvelables, en particulier l'hydraulique, l'éolien, et le solaire, gagnent en importance, soutenant les objectifs de transition écologique. En parallèle, la part des énergies fossiles diminue progressivement, illustrant l'engagement de la France à réduire sa dépendance aux sources polluantes. Cette répartition reflète une stratégie énergétique alliant sécurité et ambitions environnementales.
""")

  ###################################################################################################  
  
  
  st.write("A présent, explorons l'évolution de la production d'énergie en France entre 2018 et 2022, en distinguant les sources renouvelables (éolien, solaire, hydraulique) des sources fossiles et nucléaires.")

  
  
  
  
  
  

  df['date'] = pd.to_datetime(df['date'], errors='coerce')


  production_columns = ['thermique', 'nucleaire', 'eolien', 'solaire', 'hydraulique', 'bioenergies']
  for energy in production_columns:
      df[energy] = pd.to_numeric(df[energy], errors='coerce')


  df['year'] = df['date'].dt.year


  years = [2018, 2019, 2020, 2021, 2022]
  selected_years = st.multiselect("Sélectionnez les années pour afficher l'évolution de la production d'énergie :", years, default=years)


  df_selected_years = df[df['year'].isin(selected_years)]


  yearly_production = df_selected_years.groupby('year')[production_columns].sum().reset_index()


  colors = {
    'eolien': 'red',
    'solaire': 'skyblue',
    'hydraulique': 'orange',
    'thermique': 'Lightgreen',
    'bioenergies': 'blue',
    'nucleaire': 'green'
}


  fig_renewable = go.Figure()
  for energy in ['eolien', 'solaire', 'hydraulique']:
      fig_renewable.add_trace(go.Scatter(x=yearly_production['year'], y=yearly_production[energy],
                                         mode='lines+markers', name=energy.capitalize(), line=dict(color=colors[energy])))

  fig_renewable.update_layout(
      title="Évolution de la Production d'Énergie Renouvelable en France",
      xaxis_title='Année',
      yaxis_title='Production (en MWh)',
      xaxis=dict(dtick=1, tickfont=dict(color='black')),
      yaxis=dict(tickfont=dict(color='black')),
      title_font=dict(color='black'),
      plot_bgcolor='white',
      paper_bgcolor='white',
      legend=dict(
          title=dict(text="Type d'Énergie", font=dict(color="black")),
          font=dict(color="black"),
          bgcolor="white",
          bordercolor="black",
          borderwidth=1
    )
)


  fig_fossil = go.Figure()
  for energy in ['thermique', 'bioenergies', 'nucleaire']:
      fig_fossil.add_trace(go.Scatter(x=yearly_production['year'], y=yearly_production[energy],
                                      mode='lines+markers', name=energy.capitalize(), line=dict(color=colors[energy])))

  fig_fossil.update_layout(
      title="Évolution de la Production d'Énergie Fossile et Nucléaire en France",
      xaxis_title='Année',
      yaxis_title='Production (en MWh)',
      xaxis=dict(dtick=1, tickfont=dict(color='black')),
      yaxis=dict(tickfont=dict(color='black')),
      title_font=dict(color='black'),
      plot_bgcolor='white',
      paper_bgcolor='white',
      legend=dict(
          title=dict(text="Type d'Énergie", font=dict(color="black")),
          font=dict(color="black"),
          bgcolor="white",
          bordercolor="black",
          borderwidth=1
    )
)


  st.write("### Évolution de la Production d'Énergie Renouvelable en France (2018-2022)")
  st.plotly_chart(fig_renewable)

  st.write("### Évolution de la Production d'Énergie Fossile et Nucléaire en France (2018-2022)")
  st.plotly_chart(fig_fossil)
  
  
  st.write("Les graphiques montrent une stabilité de la production nucléaire, confirmant son rôle central dans l'approvisionnement énergétique français. En parallèle, les énergies renouvelables, notamment l'éolien et le solaire, progressent, illustrant les efforts vers une transition énergétique plus verte. Enfin, la baisse des sources fossiles reflète la stratégie de réduction de la dépendance aux énergies polluantes, alignée avec les objectifs de durabilité.")

  ###################################################################################################  
  st.write("""
Pour approfondir notre compréhension du mix énergétique, il est essentiel de ne pas seulement examiner l'évolution des différentes sources de production, mais aussi d'analyser leur impact direct sur la consommation électrique. Les graphiques aux points que nous allons explorer permettent d'étudier la **corrélation entre la consommation et la production** pour chaque type d'énergie. Ce type de visualisation met en évidence la relation entre la quantité d'énergie produite par une source donnée et sa contribution potentielle à la demande globale.
""")



  energy_types = ['eolien', 'solaire', 'nucleaire', 'thermique', 'hydraulique']


  for energy in energy_types:
      df[energy] = pd.to_numeric(df[energy], errors='coerce')


  selected_energy = st.selectbox("Sélectionnez le type d'énergie pour la corrélation avec la consommation :", energy_types)


  filtered_df = df[['consommation', selected_energy]].dropna()


  st.write(f"### Corrélation entre la consommation et la production électrique française de {selected_energy.capitalize()}")

  plt.figure(figsize=(10, 6))
  sns.scatterplot(x='consommation', y=selected_energy, data=filtered_df)
  plt.title(f"Corrélation Consommation vs {selected_energy.capitalize()}")
  plt.xlabel("Consommation (MWh)")
  plt.ylabel(f"{selected_energy.capitalize()} (MWh)")
  st.pyplot(plt)
  
  
  
  
  st.write("""
Les différents résultats nous montrent que les sources d'énergie renouvelables, comme l'éolien et le solaire, présentent une corrélation limitée avec la consommation, en raison de leur variabilité dépendante des conditions climatiques. 

Le nucléaire, en revanche, offre une production stable et corrélée à la demande, jouant un rôle clé dans le mix énergétique français. Le thermique et l'hydraulique, quant à eux, apportent la flexibilité nécessaire pour ajuster la production aux fluctuations de la demande, avec une capacité d'adaptation importante pour les pics de consommation. 

Ces observations soulignent l'importance d'une combinaison équilibrée de sources pour assurer la stabilité et la résilience du réseau électrique.
""")
 
  
  

  ###################################################################################################  
  
  
  st.write("Pour continuer, nous allons représenter l'évolution de la consommation d'énergie moyenne par mois sur plusieurs années.")

  
  
  
  
  

  df['date_heure'] = pd.to_datetime(df['date_heure'],format='%Y-%ME-%d', errors='coerce')
  df.set_index('date', inplace=True)


  df['consommation'] = pd.to_numeric(df['consommation'], errors='coerce')


  periode = st.selectbox("Sélectionnez la période pour la consommation moyenne mensuelle :", ["2012-2017", "2018-2022"])


  if periode == "2012-2017":
     df_filtered = df[(df.index.year >= 2013) & (df.index.year <= 2017)]
  else:
      df_filtered = df[(df.index.year >= 2018) & (df.index.year <= 2022)]


  consommation_mensuelle = df_filtered['consommation'].resample('M').mean()


  st.write(f"### Consommation Moyenne d'Énergie par Mois ({periode})")
  plt.figure(figsize=(12, 6))
  plt.plot(consommation_mensuelle.index, consommation_mensuelle, marker='o', linestyle='-', color='blue')
  plt.title(f"Consommation Moyenne d'Énergie par Mois ({periode})")
  plt.xlabel('Mois')
  plt.ylabel('Consommation Moyenne (MWh)')
  plt.grid(True)
  plt.xticks(rotation=45)
  st.pyplot(plt)
  
  
  
  st.write("On peut remarquer que la demande énergétique augmente de manière significative pendant les mois d'hiver en raison des besoins de chauffage, tandis qu'elle diminue pendant l'été. Cette saisonnalité impose des exigences particulières en matière de gestion de la production, car elle nécessite d'adapter la capacité de production pour répondre efficacement aux variations de la demande.")
  
  st.write("La saisonnalité étant constatée, nous allons à présent explorer la consommation moyenne d'électricité par saison. Cette approche va nous permettre d'illustrer comment la demande fluctue de manière récurrente tout au long de l'année, en réponse aux conditions climatiques et aux habitudes de consommation.")
  
  
  selected_year_conso_mean = st.selectbox("Sélectionnez l'année :", [2018, 2019, 2020, 2021, 2022], key="year_selector_seasonal")

  
  df_year_conso_mean = df[df['date_heure'].dt.year == selected_year_conso_mean]

  # Définir les saisons selon les mois
  def get_season(month):
      if month in [12, 1, 2]:
          return 'Hiver'
      elif month in [3, 4, 5]:
          return 'Printemps'
      elif month in [6, 7, 8]:
          return 'Été'
      elif month in [9, 10, 11]:
         return 'Automne'

  
  df_year_conso_mean['Saison'] = df_year_conso_mean['date_heure'].dt.month.apply(get_season)

 
  consommation_moyenne_par_saison = df_year_conso_mean.groupby('Saison')['consommation'].mean().reindex(['Hiver', 'Printemps', 'Été', 'Automne'])

  
  st.write(f"### Consommation moyenne d'électricité par saison en {selected_year_conso_mean}")
  plt.figure(figsize=(10, 6))
  consommation_moyenne_par_saison.plot(kind='bar', color='skyblue')
  plt.title(f"Consommation moyenne d'électricité par saison en {selected_year_conso_mean}")
  plt.xlabel('Saison')
  plt.ylabel('Consommation moyenne (MWh)')
  plt.xticks(rotation=45)
  plt.grid(True)
  st.pyplot(plt)
  
  
  st.write("Ces graphiques nous confirment bel et bien une saisonnalité marquée dans la consommation d'électricité, avec des photos en hiver dus aux besoins de chauffage, tandis que l'été enregistre une augmentation liée à la climatisation. Les saisons intermédiaires montrer une demande plus stable, offrant des opportunités pour la maintenance des infrastructures. ")
  
  st.write("Confirmer cette saisonnalité est essentiel pour le choix de notre modèle de prédiction de consommation électrique, car un modèle efficace doit être capable de capturer ces variations périodiques pour fournir des estimations précises. En intégrant la saisonnalité dans nos prévisions, nous pourrons anticiper avec plus de précision les pics de demande en hiver et les hausses estivales, optimisant ainsi la gestion des ressources énergétiques et entraînant les risques de surcharge du réseau. Ce type d'analyse permet de sélectionner des modèles de prévision adaptés, comme les modèles de séries temporelles qui prennent en compte les tendances et les fluctuations saisonnières, garantissant une meilleure précision dans l'estimation d'ensemble.")
  
  
  ###################################################################################################  


  st.write("Nous allons maintenant créer un graphique qui va nous permettre d'examiner en détail les variations mensuelles et de visualiser comment les tendances de consommation évoluent au fil des années. ")




  df['date_heure'] = pd.to_datetime(df['date_heure'],format='%Y-%ME-%d',errors='coerce')
  df['consommation'] = pd.to_numeric(df['consommation'], errors='coerce')


  df['year'] = df['date_heure'].dt.year
  df['month'] = df['date_heure'].dt.month


  years = list(range(2013, 2023))
  selected_years = st.multiselect("Sélectionnez les années pour afficher la consommation moyenne par mois :", years, default=[2022])


  df_filtered = df[df['year'].isin(selected_years)]


  pivot_df = df_filtered.pivot_table(values='consommation', index='month', columns='year', aggfunc='mean')


  years_text = ", ".join(map(str, selected_years))
  print("Selected Years Text:", years_text)  


  if not years_text:
      years_text = "all selected years"


  st.write(f"### Consommation Moyenne d'Énergie par Mois pour les Années Sélectionnées : {years_text}")


  plt.figure(figsize=(14, 7))
  for year in pivot_df.columns:
      plt.plot(pivot_df.index, pivot_df[year], marker='', label=year)


  plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))


  plt.title(f"Consommation Moyenne d'Énergie par Mois pour les Années : {years_text}")
  plt.xlabel('Mois')
  plt.ylabel('Consommation Moyenne (MWh)')
  plt.legend(title='Année')
  plt.grid(True)


  st.pyplot(plt)
  
  
  
  st.write("On observe ici année après année la nécessité de modéliser la consommation en tenant compte des variations saisonnières pour anticiper les périodes de forte année demande et optimiser la gestion des ressources énergétiques.")
  
  ###################################################################################################  
  st.write("Les périodes de forte consommation étant plutôt définie, nous allons examiner à présent le phasage entre la consommation et la production d'électricité au cours de l'année, en mettant en lumière les périodes de surproduction et de sous-production.")


  production_columns = ['thermique', 'nucleaire', 'eolien', 'solaire', 'hydraulique', 'bioenergies']
  for energy in production_columns:
      df[energy] = pd.to_numeric(df[energy], errors='coerce')


  df['consommation'] = pd.to_numeric(df['consommation'], errors='coerce')


  years = [2018, 2019, 2020, 2021, 2022]
  selected_year = st.selectbox("Sélectionnez l'année pour le phasage entre la consommation et la production :", years)


  df_year = df[df['date_heure'].dt.year == selected_year].copy()


  consommation_par_mois = df_year.groupby(df_year['date_heure'].dt.to_period('M'))['consommation'].sum().reset_index()
  consommation_par_mois.columns = ['mois', 'consommation_totale']


  df_year['production_totale'] = df_year[production_columns].sum(axis=1)
  production_par_mois = df_year.groupby(df_year['date_heure'].dt.to_period('M'))['production_totale'].sum().reset_index()
  production_par_mois.columns = ['mois', 'production_totale']


  phasage = pd.merge(consommation_par_mois, production_par_mois, on='mois')


  st.write(f"### Phasage entre la consommation et la production par mois en {selected_year}")
  plt.figure(figsize=(14, 8))
  plt.plot(phasage['mois'].astype(str), phasage['consommation_totale'], label='Consommation', color='blue', linestyle='-', marker='o')
  plt.plot(phasage['mois'].astype(str), phasage['production_totale'], label='Production Totale', color='green', linestyle='-', marker='o')
  plt.fill_between(phasage['mois'].astype(str), phasage['consommation_totale'], phasage['production_totale'],
                   where=(phasage['production_totale'] >= phasage['consommation_totale']), facecolor='green', alpha=0.3, label='Surproduction')
  plt.fill_between(phasage['mois'].astype(str), phasage['consommation_totale'], phasage['production_totale'],
                   where=(phasage['production_totale'] < phasage['consommation_totale']), facecolor='red', alpha=0.3, label='Sous-production')

  plt.title(f"Phasage entre la consommation et la production par mois en {selected_year}")
  plt.xlabel("Mois")
  plt.ylabel("Énergie (MWh)")
  plt.legend()
  plt.grid(True)
  plt.xticks(rotation=45)
  st.pyplot(plt)
  
  
  st.write("De 2018 à 2022, on observe une progression dans l'équilibre entre la consommation et la production d'énergie en France. Alors que les premières années montrent des déséquilibres marqués, avec des périodes de surproduction en hiver et des sous-productions en été, les dernières années (2021 et 2022) indiquent une meilleure correspondance entre production et demande. Cette évolution suggère une adaptation progressive dans la gestion de la production, probablement liée à des stratégies visant à ajuster l'offre aux besoins saisonniers, optimisant ainsi l'efficacité énergétique et la stabilité du réseau.")
  
  
  ###################################################################################################  
  
  st.write("Une visualisation cartographique va nous permettre de comprendre l'importance des dynamiques régionales dans le mix énergétique, en mettant en évidence les disparités de consommation ainsi que la contribution de chaque région en termes de production renouvelable et non renouvelable. Cela nous aidera à évaluer l'équilibre régional de l'offre et de la demande.")
  
  st.write ("### Cartographie de la Consommation ou de la Production Électrique par Région en France")
  

  gdf = gpd.read_file('regions-20180101-shp/regions-20180101.shp') 


  gdf = gdf.rename(columns={'nom': 'libelle_region'})
  gdf = gdf[['libelle_region', 'geometry']]  


  df['date_heure'] = pd.to_datetime(df['date_heure'], errors='coerce')


  year = st.selectbox("Sélectionnez l'année :", [2018, 2019, 2020, 2021, 2022])


  option = st.selectbox(
      "Sélectionnez l'indicateur à afficher sur la carte :",
      ("Consommation électrique par région",
       "Production électrique renouvelable par région",
       "Production électrique non renouvelable par région")
)


  df_year = df[df['date_heure'].dt.year == year]


  if option == "Consommation électrique par région":
      consommation_par_region = df_year.groupby('libelle_region')['consommation'].sum().reset_index()
      gdf_data = gdf.merge(consommation_par_region, how='left', on='libelle_region')
      column = 'consommation'
      title = f"Consommation Électrique par Région en France (en MW) - {year}"
      cmap = 'RdYlGn_r'

  elif option == "Production électrique renouvelable par région":
      production_renouvelable_par_region = df_year.groupby('libelle_region')[['eolien', 'solaire', 'hydraulique']].sum().sum(axis=1).reset_index()
      gdf_data = gdf.merge(production_renouvelable_par_region, how='left', on='libelle_region')
      column = 0  
      title = f"Production Électrique Renouvelable par Région en France (en MW) - {year}"
      cmap = 'YlGn'

  else:
      production_non_renouvelable_par_region = df_year.groupby('libelle_region')[['nucleaire', 'thermique', 'bioenergies']].sum().sum(axis=1).reset_index()
      gdf_data = gdf.merge(production_non_renouvelable_par_region, how='left', on='libelle_region')
      column = 0 
      title = f"Production Électrique Non Renouvelable par Région en France (en MW) - {year}"
      cmap = 'OrRd'


  def plot_map(gdf, column, title, cmap='RdYlGn_r'):
      fig, ax = plt.subplots(1, 1, figsize=(12, 10))
      norm = mcolors.Normalize(vmin=gdf[column].min(), vmax=gdf[column].max())
      gdf.plot(column=column, ax=ax, legend=True,
               cmap=cmap, norm=norm, legend_kwds={'label': title, 'orientation': "horizontal"})
      ax.set_title(title, fontsize=15)
      ax.set_axis_off()
      st.pyplot(fig)


  plot_map(gdf_data, column, title, cmap)
  
  
  st.write("Ces cartes mettent en lumière les différences régionales en matière de consommation et de production d'énergie en France. On constate que certaines régions, comme l'Île-de-France, ont une forte consommation d'énergie sans nécessairement produire de manière proportionnelle, accentuant leur dépendance vis-à-vis des autres régions. D'autre part, des régions comme Auvergne-Rhône-Alpes et Grand Est se distinguent par une production élevée, notamment en énergie non renouvelable, contribuant de manière significative à l'approvisionnement national. La production renouvelable, bien que répartie, est plus concentrée dans des zones spécifiques, souligne l'importance de politiques régionales adaptées pour optimiser l'autonomie énergétique et réduire les disparités. ")
  
  
  st.write("Pour mieux comprendre l'impact de certains facteurs régionaux sur la consommation d'énergie, nous allons examiner des cartes présentant divers indicateurs démographiques et géographiques : le nombre moyen d'habitants, la surface régionale en km², et la densité d'habitants par km². À chaque carte est associée une analyse de corrélation de Pearson, permettant d'identifier les liens entre ces variables et la consommation d'énergie. ")
  
  st.write ("### Cartographie de la Consommation par rapport à divers indicateurs démographiques et géographiques")
  


  df_gdf_2 = pd.read_csv('gdf_2.csv') 
  df_gdf_2['geometry'] = df_gdf_2['geometry'].apply(wkt.loads)
  gdf_2 = gpd.GeoDataFrame(df_gdf_2, geometry='geometry')


  year = st.selectbox("Sélectionnez l'année :", [2018, 2019, 2020, 2021, 2022], key="year_selector")


  option = st.selectbox(
      "Sélectionnez l'indicateur à afficher sur la carte :",
      ("nombre d'habitants moyen", "surface en km2", "densité d'habitants par km2"),
      key="indicator_selector"
)


  if option == "nombre d'habitants moyen":
      column = 'nombre_d_habitants'
      title = f"Carte du Nombre d'Habitants Moyen par Région en {year}"
      cmap = 'RdYlGn_r'
      label = "Nombre d'Habitants"
      y_label = "Consommation par region"
      x_label = "Nombre d'Habitants"
      plot_title = f"Relation entre Population moyenne et Consommation par région en {year}"

  elif option == "surface en km2":
      column = 'surf_km2'
      title = f"Carte de la Surface en km² par Région en {year}"
      cmap = 'Blues'
      label = "Surface en km²"
      y_label = "Consommation par region"
      x_label = "Surface (km²)"
      plot_title = f"Relation entre Surface et Consommation par région en {year}"

  else:
      column = 'densite_km2'
      title = f"Carte de la Densité d'Habitants par km² en {year}"
      cmap = 'Oranges'
      label = "Densité (habitants/km²)"
      y_label = "Consommation par region"
      x_label = "Densité d'Habitants (habitants/km²)"
      plot_title = f"Relation entre Densité d'Habitants et Consommation par région en {year}"


  def plot_map(gdf, column, title, cmap, label):
      fig, ax = plt.subplots(1, 1, figsize=(10, 10))
      norm = mcolors.Normalize(vmin=gdf[column].min(), vmax=gdf[column].max())
      gdf.plot(column=column, ax=ax, legend=True, cmap=cmap, norm=norm,
               legend_kwds={'label': label, 'orientation': "vertical"})
      ax.set_title(title, fontsize=15)
      ax.set_axis_off()
      st.pyplot(fig)


  plot_map(gdf_2, column, title, cmap, label)


  if column in gdf_2.columns and 'Consommation par region' in gdf_2.columns:
      corr_coef, p_value = pearsonr(gdf_2[column], gdf_2['Consommation par region'])

 
      st.write(f"### Coefficient de corrélation de Pearson : {corr_coef:.2f}")
      st.write(f"### P-value : {p_value:.2e}")

  
      H0 = "Les variables sélectionnées ne sont pas corrélées à la consommation."
      H1 = "Les variables sélectionnées sont corrélées à la consommation."

   
      alpha = 0.05
      if p_value < alpha:
          st.write(f"Il y a suffisamment de preuves pour rejeter H0. {H1}")
      else:
          st.write(f"Nous n'avons pas suffisamment de preuves pour rejeter H0. {H0}")

  
      plt.figure(figsize=(8, 6))
      sns.regplot(x=column, y='Consommation par region', data=gdf_2)
      plt.title(plot_title)
      plt.xlabel(x_label)
      plt.ylabel(y_label)
      plt.grid(True)
      st.pyplot(plt)
  else:
      st.write("Les données nécessaires pour le calcul de la corrélation ne sont pas disponibles.")
  
  
  
  st.write("Ces graphiques révèlent des insights intéressants sur la consommation énergétique régionale en France. Les cartes montrent que la consommation d’énergie est souvent corrélée avec des facteurs démographiques, tels que le nombre moyen d’habitants et la densité de population. Par exemple, les régions densément peuplées, comme l'Île-de-France, affichent une forte consommation, ce qui confirme une demande énergétique élevée en lien avec la concentration urbaine.")

  st.write("En revanche, l'analyse de la surface régionale montre une faible corrélation avec la consommation, suggérant que la taille géographique n’a pas un impact significatif. Cela met en évidence que ce n'est pas l'étendue territoriale mais bien les caractéristiques de peuplement et d'activité économique qui influencent la demande énergétique.")
  
  #####################################################################################################  
  
  st.write("Dans la suite du projet, nous nous sommes concentrés sur deux régions : l'Île-de-France et la Bourgogne-Franche-Comté. Nous allons maintenant examiner comment les variations de température influencent la consommation d'énergie par habitant dans ces régions spécifiques. Les graphiques suivants illustrent la relation entre la consommation énergétique par habitant et la température moyenne quotidienne en 2022, permettant de mieux comprendre l'impact des conditions climatiques sur la demande énergétique. ")
  st.image("IDFBFC.png")

  region=df_ml['libelle_region'].unique()

  selection_region=st.selectbox(label="Région",options= region)
  annee=df_ml['Année'].unique()

  selection_a=st.selectbox(label="Année",options= annee)
  filtered_data2 = df_ml.loc[df_ml['libelle_region'] ==  selection_region]
  filtered_data2=filtered_data2[filtered_data2['Année']==selection_a]

  
  consoM=filtered_data2.groupby(pd.Grouper(key='date_heure_simplifiée', freq='W'))[['consommation','tmoy','tmax','tmin']].mean()



  st.write ("### Existe t'il un lien de corrélation entre la consommation électrique et les température moyenne ? ")
  st.write("Nous allons procéder à un de corrélation de Pearsonr pour affirmer ou bien infirmer nos hypothèses suivantes :")
  st.write("H0 : Les températures ne sont pas corrélées à la consommation.")
  st.write("H1 : Les températures sont corrélées à la consommation.")

  corr, p_value = pearsonr(consoM["consommation"], consoM["tmoy"])
  st.write(f"Le Coefficient de corrélation de Pearson est: {corr:.2f}")
  st.write(f"La P-value est : {p_value:.2e}")

  
  H0 = "La température moyenne n'est pas corrélée à la consommation."
  H1 = "La température moyenne est corrélée à la consommation."


  alpha = 0.05
  if p_value < alpha:
      st.write(f"Il y a suffisamment de preuves pour rejeter H0. {H1}")
  else:
      st.write(f"Nous n'avons pas suffisamment de preuves pour rejeter H0. {H0}")




  st.write(f"### Consommation d'énergie par habitant en fonction de la température moyenne en {selection_region} pour l'année {selection_a}")
 
  
     
  plt.figsize=(5, 5) 
  sns.relplot(x = "tmoy", y = "consommation", kind = "line", data = consoM)
  plt.title ("Consommation moyenne par semaine en fonction des températures moyenne")
  plt.xlabel("Température moyenne")
  plt.ylabel("Consommation moyenne")
  st.pyplot(plt)
  
  
  
  
  st.write("Nous avons ici l'illustration d'une relation inverse entre la température moyenne quotidienne et la consommation d'énergie par habitant dans les régions analysées.Plus les températures augmentent, moins la consommation d'énergie par habitant est élevée, ce qui est cohérent avec une réduction des besoins de chauffage durant les périodes chaudes. En revanche, lors des périodes froides, la consommation augmente de manière notable, traduisant un recours accru aux dispositifs de chauffage. Cette tendance saisonnière suggère que la température est un facteur clé de la demande énergétique.")
  
  

  #####################################################################################################
  
  st.write("Pour approfondir l'impact des différentes variables sur la consommation d'énergie, il est essentiel d'examiner comment celle-ci évolue en fonction des secteurs d’activité. En effet, les besoins énergétiques varient considérablement entre l’agriculture, l’industrie, et le secteur tertiaire, chacun ayant des caractéristiques et des usages de l’électricité spécifiques. ")
  
  #selection_annee=st.multiselect(label="Année",options= annee)

  filtered_data = df_ml.loc[df_ml['libelle_region'] ==  selection_region]
  #filtered_data=filtered_data[filtered_data['Année'].isin (selection_annee)]


  st.write ("### Consommation Electrique Moyenne par Secteurs d'Activités")
  

  secteur=filtered_data.groupby(['Année','libelle_region']).agg({'conso_totale_mwh_tertiaire_annuel':'mean','conso_moyenne_mwh_tertiaire_annuel':'mean','conso_totale_mwh_agriculture_annuel':'mean','conso_moyenne_mwh_agriculture_annuel':'mean','conso_totale_mwh_industrie_annuel':'mean','conso_moyenne_mwh_industrie_annuel':'mean'}).reset_index()
  
  plt.figure(figsize=(14, 8))
  plt.plot(secteur['Année'].astype(str), secteur['conso_moyenne_mwh_agriculture_annuel'], label='Agriculture', color='blue', linestyle='-', marker='o')
  plt.plot(secteur['Année'].astype(str),secteur['conso_moyenne_mwh_industrie_annuel'], label='Industrie', color='r', linestyle='-', marker='o')
  plt.plot(secteur['Année'].astype(str),secteur['conso_moyenne_mwh_tertiaire_annuel'], label='Tertiaire', color='green', linestyle='-', marker='o')
  plt.title("Consommation electrique moyenne par secteur d'activité")
  plt.xlabel("Année")
  plt.ylabel("Consommation (MWh)")
  plt.legend()
  st.pyplot(plt)
  
  
  st.write("On peut constater ici des différences de consommation électrique entre les secteurs. L'industrie, avec la consommation la plus élevée, montre une légère baisse depuis 2018, suggérant des efforts d'efficacité ou des fluctuations économiques. Le secteur tertiaire est plus stable, tandis que l'agriculture, avec la plus faible consommation, reste constante. Ces variations soulignent l'importance d'adapter les stratégies énergétiques selon les spécificités de chaque secteur.")
  
  
  st.markdown("""
### Construction du DataFrame Final

Suite à l'analyse visuelle des tendances de consommation, nous passons maintenant à une étape cruciale : **la construction de notre DataFrame final**, qui servira de base pour nos futures analyses prédictives. Ce DataFrame intègre plusieurs variables essentielles, incluant :

- **Les effets saisonniers** observés dans les variations de consommation,
- **La température moyenne**, un facteur clé influençant la demande énergétique,
- **La production totale d'électricité**, pour assurer une vue d'ensemble,
- **Les différentes sources d'énergie** (nucléaire, renouvelables, fossiles) et leur contribution respective,
- **La densité de population**, en particulier dans les régions à forte consommation,
- **La répartition par secteurs d'activité** (résidentiel, industriel, tertiaire),
- **Les spécificités régionales**, telles que les disparités de production et de consommation.

En réunissant ces données, nous constituons une base solide qui nous permettra de mieux comprendre les dynamiques de consommation d'électricité et de poser les fondations de futures prévisions énergétiques fiables.

Dans la section suivante, nous nous appuierons sur ce DataFrame pour construire nos modèles prédictifs.
""")

  #####################################################################################################

if page == pages[3] :
    
  st.image("Energie_13.jpg")  
  
  st.write("### Modélisation")  

  st.write ("Le jeu de donnée que nous allons utilisées pour le machine learning est le suivant :")
  st.dataframe(df_ml.head())

  st.write ('**Analyse des series temporelles**')

  st.write("Pour cette première analyse temporelle mensuelle, nous nous penchons sur la consommation électrique en Île-de-France et en Bourgogne-Franche-Comté. Depuis le début de notre étude, nous avons clairement identifié une saisonnalité marquée dans la consommation électrique. L'objectif va donc être d'explorer les tendances saisonnières et de stationnarité pour identifier le modèle le plus adapté pour prédire la consommation future. Voici une présentation des graphiques et une interprétation de chaque étape.")
  
  ###########################

  model_choisi=st.selectbox(label="Model",options= {'Sarimax','Prophet','Comparaison SARIMAX/PROPHET'})

  region=df_ml['libelle_region'].unique()
  selection_region=st.selectbox(label="Région",options= region)
  

  filtered_data = df.loc[df['libelle_region'] ==  selection_region]

  filtered_data['date'] = pd.to_datetime(filtered_data['date'])

  sarima_data=filtered_data.groupby(pd.Grouper(key='date', freq='M'))['consommation'].mean()
   

  X_train_sarima = sarima_data[:-24]
  y_test_sarima = sarima_data[-24:]
   

  
  decoupage = plt.figure(figsize=(10, 4))
  plt.plot(X_train_sarima, label="Entraînement", color="blue")
  plt.plot(y_test_sarima,label="Test", color="orange")
  plt.xlabel('Année')
  plt.ylabel('Consommation (MWh)')
  plt.title(f'Consommation Électrique Moyenne par Mois en {selection_region}')
  plt.legend()
  
    
  res_add = seasonal_decompose(X_train_sarima, model='additive')
  fig_additive = res_add.plot()
 
 
  res_mul = seasonal_decompose(X_train_sarima, model='multiplicative')
  fig_multi = res_mul.plot()


  train_log = np.log(X_train_sarima)
  
  temp=plt.figure(figsize=(10, 4))
  plt.plot(train_log)
  plt.title(f'Série Temporelle Logarithmique - {selection_region}') 



  train_log_diff1 = train_log.diff().dropna()

  
  fig_diff1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
  train_log_diff1.plot(ax=ax1, title="Série différenciée d'ordre 1")
  autocorrelation_plot(train_log_diff1, ax=ax2)
 
  
  train_log_diff1_12 = train_log_diff1.diff(12).dropna()

  fig_diff12, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
  train_log_diff1_12.plot(ax=ax1, title="Série doublement différenciée (Ordre 1 & 12)")
  autocorrelation_plot(train_log_diff1_12, ax=ax2)
  

  max_lags_acf = min(20, len(train_log_diff1_12) - 1) 
  max_lags_pacf = min(10, len(train_log_diff1_12) // 2 - 1)  

  fig_acf, ax = plt.subplots(figsize=(8, 4))
  plot_acf(train_log_diff1_12, lags=max_lags_acf, ax=ax)
 

  fig_pacf, ax = plt.subplots(figsize=(8, 4))
  plot_pacf(train_log_diff1_12, lags=max_lags_pacf, ax=ax)
 
 
   
  
  ###########################


#SARIMAX

   
  pickle_in= open('sarimax_model.pkl', 'rb') 
  loaded_results = pickle.load(pickle_in)

  pickle_in_BFC= open('sarimax_model_BFC.pkl', 'rb') 
  loaded_results_BFC = pickle.load(pickle_in_BFC)


  start=len(X_train_sarima)
  end=len(X_train_sarima)+len(y_test_sarima)-1
 
  y_pred_sarima =np.exp(loaded_results.predict(start = start, end = end))
  
  
  y_pred_sarima_BFC =np.exp(loaded_results_BFC.predict(start = start, end = end))

  y_test_a=y_test_sarima[:12]
  
  

  end_annuel=len(X_train_sarima)+len(y_test_a)-1

  #Prediction sur l'ensemble du jeu de test

  y_pred_sarima_annuelle =np.exp(loaded_results.predict(start = y_test_a.index[0], end = y_test_a.index[-1]))

  y_pred_sarima_annuelle_BFC =np.exp(loaded_results_BFC.predict(start = y_test_a.index[0], end = y_test_a.index[-1]))
  #Prophet------------------------------------------------------------------------------------------------------------------

  #@title Regroupement par mois de la conso et temperature moyenne
   
  filtered_data_prophet = df_ml.loc[df_ml['libelle_region'] ==  selection_region]

  consoMensuelle=filtered_data_prophet.groupby(pd.Grouper(key='date_heure_simplifiée', freq='M'))[['consommation','tmoy','tmax','tmin']].mean()


  #@title Prepa DateFrame Prophet MULTIVARIE

  #Rename the columns to 'ds' and 'y'
  consoMensuelle = consoMensuelle.reset_index().rename(columns={'date_heure_simplifiée':'ds', 'consommation':'y'})


  # @title Separation jeu de train et test

  df_train = consoMensuelle.loc[consoMensuelle["ds"]<"2021-12-31"]
  df_test  = consoMensuelle.loc[consoMensuelle["ds"]>="2021-12-31"]

  from prophet import Prophet

  model_prophet= Prophet(seasonality_mode='multiplicative', yearly_seasonality=3)

  model_prophet.add_regressor('tmoy')
  model_prophet.add_regressor('tmax')
  model_prophet.add_regressor('tmin')

  model_prophet.fit(df_train)


  # @title Affichage des dates futures concerné par la prévision
  future = model_prophet.make_future_dataframe(periods=12,freq='M')
  future['tmoy']=consoMensuelle['tmoy']
  future['tmax']=consoMensuelle['tmax']
  future['tmin']=consoMensuelle['tmin']
  future.tail()

  forecast = model_prophet.predict(future)






  from prophet.diagnostics import cross_validation, performance_metrics

  # Perform cross-validation to generate forecasts with a 'cutoff' column
  df_cv = cross_validation(model_prophet, initial='1000 days', period='30 days', horizon = '365 days')
  cutoffs = pd.to_datetime(['2020-10-31','2020-11-30'])
  df_cv2 = cross_validation(model_prophet, cutoffs=cutoffs, horizon='365 days', period='30 days')



 
  #@title Metrique de performance PROPHET a divers Horizon

  from prophet.diagnostics import performance_metrics
  df_performance = performance_metrics(df_cv)

  df_performance2 = performance_metrics(df_cv2)
 

 
  perf=df_performance.loc[df_performance['horizon'] == '365 days']   

  figval=plot_cross_validation_metric(df_cv,metric='mae')
  figval1=plot_cross_validation_metric(df_cv,metric='rmse')



  score_idf_24M = r2_score(y_test_sarima, y_pred_sarima)
  mae_idf = mean_absolute_error(y_test_sarima, y_pred_sarima)
  mse_idf= mean_squared_error(y_test_sarima, y_pred_sarima)
  rmse_idf= np.sqrt(mean_squared_error(y_test_sarima, y_pred_sarima))
  data_idf_24M= {'Metric': ['MAE', 'MSE', 'RMSE'], 'Durée' : ['24 mois','24 mois','24 mois'],  'SARIMA': [mae_idf, mse_idf, rmse_idf]}


  score_bfc_24M = r2_score(y_test_sarima, y_pred_sarima_BFC)
  mae_bfc= mean_absolute_error(y_test_sarima, y_pred_sarima_BFC)
  mse_bfc= mean_squared_error(y_test_sarima, y_pred_sarima_BFC)
  rmse_bfc= np.sqrt(mean_squared_error(y_test_sarima, y_pred_sarima_BFC))
  data_bfc_24M= {'Metric': ['MAE', 'MSE', 'RMSE'], 'Durée' : ['24 mois','24 mois','24 mois'],  'SARIMA': [mae_bfc, mse_bfc, rmse_bfc]}



    

  score_idf = r2_score(y_test_a, y_pred_sarima_annuelle)
  mae_idf = mean_absolute_error(y_test_a, y_pred_sarima_annuelle)
  mse_idf= mean_squared_error(y_test_a, y_pred_sarima_annuelle)
  rmse_idf= np.sqrt(mean_squared_error(y_test_a, y_pred_sarima_annuelle))
  data_idf= {'Metric': ['MAE', 'MSE', 'RMSE'], 'Durée' : ['12 mois','12 mois','12 mois'],  'SARIMA': [mae_idf, mse_idf, rmse_idf]}

  score_bfc = r2_score(y_test_a, y_pred_sarima_annuelle_BFC)
  mae_bfc= mean_absolute_error(y_test_a, y_pred_sarima_annuelle_BFC)
  mse_bfc= mean_squared_error(y_test_a, y_pred_sarima_annuelle_BFC)
  rmse_bfc= np.sqrt(mean_squared_error(y_test_a, y_pred_sarima_annuelle_BFC))
  data_bfc= {'Metric': ['MAE', 'MSE', 'RMSE'], 'Durée' : ['12 mois','12 mois','12 mois'],  'SARIMA': [mae_bfc, mse_bfc, rmse_bfc]}

#Dataviz SARIMAX------------------------------------------------------------------------------------


  end_pred=len(X_train_sarima)+len(y_test_sarima)-1

  import datetime
  pred = np.exp(loaded_results.predict(start=end, end=end+12))#Prédiction et passage à l'exponentielle
  pred_res = loaded_results.get_prediction(start=end, end=end+12)

 
  pred_ci = pred_res.conf_int(alpha=0.05) #Intervalle de confiance
  pred_ci=np.exp(pred_ci)
  pred_ci.index = pd.date_range(pred_ci.index[0], periods=len(pred_ci), freq='M')

  pred_low=pred_ci['lower consommation']
  pred_high=pred_ci['upper consommation']

  


 


  pred_BFC = np.exp(loaded_results_BFC.predict(start=end, end=end+12))#Prédiction et passage à l'exponentielle
  pred_res_BFC = loaded_results_BFC.get_prediction(start=end, end=end+12)



  pred_ci_BFC = pred_res_BFC.conf_int(alpha=0.05) #Intervalle de confiance
  pred_ci_BFC=np.exp(pred_ci_BFC)
  pred_ci_BFC.index = pd.date_range(pred_ci_BFC.index[0], periods=len(pred_ci_BFC), freq='M')

  pred_low_BFC=pred_ci_BFC['lower consommation']
  pred_high_BFC=pred_ci_BFC['upper consommation']


  sarima_idf= plt.figure(figsize=(10,8))
  plt.plot(X_train_sarima, label='Données d\'entrainement')
  plt.plot(y_test_sarima, label='Données de test')
  plt.axvline(x=y_test_sarima.index[-1], color='red', linestyle='--', label='Début des Prédictions')
  plt.axvline(x=X_train_sarima.index[-1], color='green', linestyle='--', label='Début des Test')
  plt.plot(y_pred_sarima, label='Prédiction sur les données de test')
  plt.plot(pred, label='Prédiction sur les données futures')
  plt.fill_between(pred_ci.index, pred_low, pred_high, alpha=0.5, color="y", label="Intervalle de confiance")
  plt.ylim(min(X_train_sarima.min(),y_test_sarima.min()),max(X_train_sarima.max(),y_test_sarima.max()))
  plt.xlabel('Date')
  plt.ylabel('Consommation')
  plt.legend()
  plt.title('Prédiction de la consommation électrique')


  sarima_bfc= plt.figure(figsize=(10,8))
  plt.plot(X_train_sarima, label='Données d\'entrainement')
  plt.plot(y_test_sarima, label='Données de test')
  plt.axvline(x=y_test_sarima.index[-1], color='red', linestyle='--', label='Début des Prédictions')
  plt.axvline(x=X_train_sarima.index[-1], color='green', linestyle='--', label='Début des Test')
  plt.plot(y_pred_sarima_BFC, label='Prédiction sur les données de test')
  plt.plot(pred_BFC, label='Prédiction sur les données futures')
  plt.fill_between(pred_ci_BFC.index, pred_low_BFC, pred_high_BFC, alpha=0.5, color="y", label="Intervalle de confiance")
  plt.ylim(min(X_train_sarima.min(),y_test_sarima.min()),max(X_train_sarima.max(),y_test_sarima.max()))
  plt.xlabel('Date')
  plt.ylabel('Consommation')
  plt.legend()
  plt.title('Prédiction de la consommation électrique')





  fig = model_prophet.plot_components(forecast)


  fig1 = model_prophet.plot(forecast)
  plt.xlabel('Date')
  plt.ylabel('Consommation')
  plt.title('Prédiction et consommation électrique')
  plt.legend()

 #Prédiction sur un intervalle de 15 mois idf
  pred2 = np.exp(loaded_results.predict(start=end, end=end+15))#Prédiction et passage à l'exponentielle
  pred_res2 = loaded_results.get_prediction(start=end, end=end+15)


  pred_ci2 = pred_res2.conf_int(alpha=0.05) #Intervalle de confiance
  pred_ci2=np.exp(pred_ci2)
  pred_ci2.index = pd.date_range(pred_ci2.index[0], periods=len(pred_ci2), freq='M')
 
  pred_low2=pred_ci2['lower consommation']
  pred_high2=pred_ci2['upper consommation']


  
 #Prédiction sur un intervalle de 15 mois bfc
  pred2_BFC = np.exp(loaded_results_BFC.predict(start=end, end=end+15))#Prédiction et passage à l'exponentielle
  pred_res2_BFC = loaded_results_BFC.get_prediction(start=end, end=end+15)


  pred_ci2_BFC = pred_res2_BFC.conf_int(alpha=0.05) #Intervalle de confiance
  pred_ci2_BFC=np.exp(pred_ci2_BFC)
  pred_ci2_BFC.index = pd.date_range(pred_ci2_BFC.index[0], periods=len(pred_ci2_BFC), freq='M')
 
  pred_low2_BFC=pred_ci2_BFC['lower consommation']
  pred_high2_BFC=pred_ci2_BFC['upper consommation']




  
  pickle_in_idf_2= open('idf.pkl', 'rb') 
  results_idf_2 = pickle.load(pickle_in_idf_2)

  
  df_sarima_idf['date'] = pd.to_datetime(df_sarima_idf['Date'])
  df_sarima_idf=df_sarima_idf.groupby(pd.Grouper(key='date', freq='M'))['consommation'].mean()

 
  X_train_sarima_idf=df_sarima_idf[:-24]
  y_test_sarima_idf=df_sarima_idf[-24:]

  start=len(X_train_sarima_idf)
  end=len(X_train_sarima_idf)+len(y_test_sarima_idf)-1
 
  y_pred_sarima_idf_2 =np.exp(results_idf_2.predict(start = start, end = end))

    
  
  y_test_idf_2=y_test_sarima_idf[:12] 

  end_annuel_idf=len(X_train_sarima_idf)+len(y_test_idf_2)-1

    
  #Prediction sur l'ensemble du jeu de test

  y_pred_sarima_annuelle_idf_2 =np.exp(results_idf_2.predict(start = y_test_idf_2.index[0], end = y_test_idf_2.index[-1]))

  pred_idf_2 = np.exp(results_idf_2.predict(start=end, end=end+12))#Prédiction et passage à l'exponentielle
  pred_res_idf2 = results_idf_2.get_prediction(start=end, end=end+12)
  
  pred_ci_IDF2 = pred_res_idf2.conf_int(alpha=0.05) #Intervalle de confiance
  pred_ci_IDF2=np.exp(pred_ci_IDF2)
  pred_ci_IDF2.index = pd.date_range(pred_ci_IDF2.index[0], periods=len(pred_ci_IDF2), freq='M')
 
  pred_low2_IDF2=pred_ci_IDF2['lower consommation']
  pred_high2_IDF2=pred_ci_IDF2['upper consommation']




  
  score_idf_2 = r2_score(y_test_idf_2, y_pred_sarima_annuelle_idf_2)
  mae_idf_2= mean_absolute_error(y_test_idf_2, y_pred_sarima_annuelle_idf_2)
  mse_idf_2= mean_squared_error(y_test_idf_2, y_pred_sarima_annuelle_idf_2)
  rmse_idf_2= np.sqrt(mean_squared_error(y_test_idf_2, y_pred_sarima_annuelle_idf_2))
  data_idf_2= {'Metric': ['MAE', 'MSE', 'RMSE'], 'Durée' : ['12 mois','12 mois','12 mois'],  'SARIMA': [mae_idf_2, mse_idf_2, rmse_idf_2]}


  y_test_idf_2_1=y_test_sarima_idf[-16:] 


  score_idf_2_1 = r2_score(y_test_idf_2_1, pred2)
  mae_idf_2_1= mean_absolute_error(y_test_idf_2_1, pred2)
  mse_idf_2_1= mean_squared_error(y_test_idf_2_1, pred2)
  rmse_idf_2_1= np.sqrt(mean_squared_error(y_test_idf_2_1, pred2))
  data_idf_2_1= {'Metric': ['MAE', 'MSE', 'RMSE'], 'Durée' : ['12 mois','12 mois','12 mois'],  'SARIMA': [mae_idf_2_1, mse_idf_2_1, rmse_idf_2_1]}

       
  comparaison_sarima_idf= plt.figure(figsize=(10,8))
  plt.plot( y_test_idf_2_1, label='Donnée réel')
  #plt.plot(y_pred_sarima_idf_2,label='prediction SARIMA',color='red')
  plt.plot(pred2, label='Prédiction avec les données initiales')
  plt.plot(pred_idf_2, label='Prédiction Future',color='green')
  plt.axvline(x=y_test_idf_2_1.index[-1], color='red', linestyle='--', label='Début des Prédictions')
  plt.fill_between(pred_ci2.index, pred_low2, pred_high2, alpha=0.5, color="y", label="Intervalle de confiance sur prédiction intiale")
  plt.fill_between(pred_ci_IDF2.index, pred_low2_IDF2, pred_high2_IDF2, alpha=0.5, color="g", label="Intervalle de confiance sur Prédiction Future")
  plt.legend()
  plt.title("Prédiction SARIMA dans le temps")



  pickle_in_bfc_2= open('bfc.pkl', 'rb') 
  results_bfc_2 = pickle.load(pickle_in_bfc_2)

  
  df_sarima_bfc2['date'] = pd.to_datetime(df_sarima_bfc2['Date'])
  df_sarima_bfc2=df_sarima_bfc2.groupby(pd.Grouper(key='date', freq='M'))['consommation'].mean()

 
  X_train_sarima_bfc2=df_sarima_bfc2[:-24]
  y_test_sarima_bfc2=df_sarima_bfc2[-24:]

  startbfc=len(X_train_sarima_bfc2)
  endbfc=len(X_train_sarima_bfc2)+len(y_test_sarima_bfc2)-1
 
  y_pred_sarima_bfc_2 =np.exp(results_bfc_2.predict(start = startbfc, end = endbfc))

    
  
  y_test_bfc_2=y_test_sarima_bfc2[:12] 

  end_annuel_bfc2=len(X_train_sarima_bfc2)+len(y_test_bfc_2)-1

    
  #Prediction sur l'ensemble du jeu de test

  y_pred_sarima_annuelle_bfc_2 =np.exp(results_bfc_2.predict(start = y_test_bfc_2.index[0], end = y_test_bfc_2.index[-1]))

  pred_BFC_2 = np.exp(results_bfc_2.predict(start=end, end=end+12))#Prédiction et passage à l'exponentielle
  pred_res_bfc2 = results_bfc_2.get_prediction(start=end, end=end+12)
  
  pred_ci_bfc2 = pred_res_bfc2.conf_int(alpha=0.05) #Intervalle de confiance
  pred_ci_bfc2=np.exp(pred_ci_bfc2)
  pred_ci_bfc2.index = pd.date_range(pred_ci_bfc2.index[0], periods=len(pred_ci_bfc2), freq='M')
 
  pred_low2_bfc2_1=pred_ci_bfc2['lower consommation']
  pred_high2_bfc2_1=pred_ci_bfc2['upper consommation']


    
  
  score_bfc_2 = r2_score(y_test_bfc_2, y_pred_sarima_annuelle_bfc_2)
  mae_bfc_2= mean_absolute_error(y_test_bfc_2, y_pred_sarima_annuelle_bfc_2)
  mse_bfc_2= mean_squared_error(y_test_bfc_2, y_pred_sarima_annuelle_bfc_2)
  rmse_bfc_2= np.sqrt(mean_squared_error(y_test_bfc_2, y_pred_sarima_annuelle_bfc_2))
  data_bfc_2= {'Metric': ['MAE', 'MSE', 'RMSE'], 'Durée' : ['12 mois','12 mois','12 mois'],  'SARIMA': [mae_bfc_2, mse_bfc_2, rmse_bfc_2]}


  y_test_bfc_2_1=y_test_sarima_bfc2[-16:] 


  score_bfc_2_1 = r2_score(y_test_bfc_2_1, pred2_BFC)
  mae_bfc_2_1= mean_absolute_error(y_test_bfc_2_1, pred2_BFC)
  mse_bfc_2_1= mean_squared_error(y_test_bfc_2_1, pred2_BFC)
  rmse_bfc_2_1= np.sqrt(mean_squared_error(y_test_bfc_2_1, pred2_BFC))
  data_bfc_2_1= {'Metric': ['MAE', 'MSE', 'RMSE'], 'Durée' : ['12 mois','12 mois','12 mois'],  'SARIMA': [mae_bfc_2_1, mse_bfc_2_1, rmse_bfc_2_1]}
       
  comparaison_sarima_bfc= plt.figure(figsize=(10,8))
  plt.plot( y_test_bfc_2_1, label='Donnée réel')
  plt.plot(pred2_BFC, label='Prédiction avec les données initiales')
  plt.plot(pred_BFC_2, label='Prédiction sur les données futures',color='green')
  plt.axvline(x=y_test_bfc_2_1.index[-1], color='red', linestyle='--', label='Début des Prédictions')
  plt.fill_between(pred_ci2_BFC.index, pred_low2_BFC, pred_high2_BFC, alpha=0.5, color="y", label="Intervalle de confiance sur prédiction intiale")
  plt.fill_between(pred_ci_bfc2.index, pred_low2_bfc2_1, pred_high2_bfc2_1, alpha=0.5, color="g", label="Intervalle de confiance sur Prédiction Future")
  plt.title("Prediction SARIMA")



    

 
  if selection_region=="Île-de-France" :
    loaded_results=loaded_results
    score=score_idf_24M
    data=data_idf_24M
    sarima=sarima_idf
    resultat2=results_idf_2.summary()
    score2=score_idf_2
    data2=data_idf_2
    sarima2=comparaison_sarima_idf
    score2_1=score_idf_2_1
    data2_1=data_idf_2_1
    score_12=score_idf
    data_12=data_idf
 
     
  elif selection_region=="Bourgogne-Franche-Comté" :
    loaded_results=loaded_results_BFC
    score=score_bfc_24M
    data=data_bfc_24M
    sarima=sarima_bfc
    resultat2=results_bfc_2.summary()
    score2=score_bfc_2
    data2=data_bfc_2
    sarima2=comparaison_sarima_bfc
    score2_1=score_bfc_2_1
    data2_1=data_bfc_2_1
    score_12=score_bfc
    data_12=data_bfc
 

  





  if model_choisi=="Prophet" :
    st.write('Les résultats des 5 premières estimation future sont :',forecast.head())

    st.subheader(f"Validation croisée {selection_region}")

    #st.dataframe(df_cv2)
    #st.write(df_performance2)

    #st.dataframe(df_cv)
    
    st.write (df_performance.head(12))
    st.markdown("Nous avons ensuite calculé les métriques MSE, RMSE, MAE, MAPE, et SMAPE pour évaluer la précision du modèle. Ces métriques mesurent l'écart entre les prédictions et les valeurs réelles à différents horizons (31 jours, 61 jours, 90 jours, etc.), et l'intervalle de confiance (coverage) indique la proportion de prédictions correctement couvertes par les prévisions.Les résultats montrent que le modèle devient moins précis à mesure que l'horizon de prédiction s'allonge, ce qui est une tendance attendue dans les prévisions de séries temporelles.")

    #st.pyplot(figval)
    #st.pyplot(figval1)
    # Python
    st.subheader(f"Visualisation des composantes du modèle Prophet {selection_region}")
    st.pyplot(fig)
    st.write("""
    Les graphiques montrent :
    ● Tendance (trend) : La consommation énergétique montre une baisse progressive entre 2018 et 2023, suggérant une rationalisation à long terme.
    ● Saisonnalité annuelle (yearly) : Les pics de consommation en hiver et les baisses en été sont bien capturés, reflétant les comportements de chauffage et de climatisation.
    ● Régresseurs multiplicatifs : Les températures influencent clairement la consommation, avec des variations visibles selon les saisons et les conditions climatiques extrêmes.
             """)
    
    st.subheader(f"Visualisation des prédictions du modèle Prophet  {selection_region}")
    st.pyplot(fig1)
    st.markdown(' Le graphique montre que le modèle capture efficacement les cycles saisonniers, avec des pics en hiver et des baisses en été. Les données réelles (points noirs) sont proches des prévisions, démontrant la précision du modèle.')

  elif model_choisi=="Sarimax":
    st.subheader(f"Consommation Électrique Moyenne par Mois pour {selection_region} (Entraînement vs Test)")
    st.pyplot(decoupage)    
    st.write("Nous avons ici la consommation électrique moyenne par mois, divisée en périodes d'entraînement et de test. La période d'entraînement couvre les années antérieures, tandis que la période de test se concentre sur les 24 derniers mois. Ce découpage permet de visualiser la dynamique historique de la série temporelle et de mieux comprendre les tendances saisonnières et les variations annuelles. On observe des photos de consommation régulières en hiver, reflétant une demande accumulée en chauffage.")
    
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        
      st.image("Energie_18.png", use_column_width="auto")
    
    
    st.write("Nous utilisons la décomposition additive et multiplicative pour séparer la série temporelle en trois composants : tendance, saisonnalité, et résidu.")

    st.subheader("Décomposition Additive")
    st.pyplot(fig_additive)
  
    st.write("""
     Ce modèle suppose que les variations de saisonnalité sont constantes dans le temps.
     Le graphique décomposé montre une tendance générale à la baisse, une saisonnalité annuelle marquée par des pics récurrents, et un résidu relativement stable.
     Ceci dit, le résidu ne semble pas aléatoire, nous allons donc voir par la suite si notre modéle correspond à un modèle multiplicatif.""")
    
    st.subheader("Décomposition Multiplicative")
    st.pyplot(fig_multi)        
    st.write("""Dans ce modèle, la saisonnalité est proportionnelle à la tendance. 
             La partie résiduelle est considérée comme un bruit blanc.
             Nous pouvons identifier que nous ne sommes pas dans le cas d'une tendance croissante linéairement dépendante du temps.
             Nous avons un cycle saisonnier de 12 mois.""")
    
    st.subheader("Série Temporelle avec Transformation Logarithmique")
    st.pyplot(temp)
    st.write("""La transformation logarithmique de la série est appliquée pour stabiliser la variance, réduisant ainsi l’amplitude des pics de consommation en hiver.
             Cette transformation rend la série plus homogène, ce qui est favorable pour l'application de modèles SARIMA.""")
    

    st.subheader("Autocorrélation - Différenciation Ordre 1")
    st.pyplot(fig_diff1)    
    st.write("""La différenciation d'ordre 1 montre la série temporelle après avoir supprimé la tendance linéaire.
             Cette transformation vise à rendre la série stationnaire en éliminant les effets de tendance. 
             Le graphique d'autocorrélation montre une saisonnalité résiduelle, indiquant que la série n'est pas totalement stationnaire.""")
    
    st.subheader("Autocorrélation - Différenciation Saisonnière (Ordre 12)")
    st.pyplot(fig_diff12)    
    st.write("""Après une différenciation saisonnière (période de 12 mois), le graphique montre que la série est devenue plus stationnaire.
             Ce type de différenciation est crucial pour capturer la structure saisonnière, surtout pour des données mensuelles.
             Le test ADF (Augmented Dickey-Fuller) confirme également la stationnarité de la série après cette transformation.""")
    
    
    st.subheader("Test de Stationnarité (ADF)")
    st.write("Le test ADF est utilisé pour vérifier si la série est stationnaire. Une P-value inférieure à 0.05 indique que la série est stationnaire après les transformations appliquées.")


    _, p_value, _, _, _, _ = sm.tsa.stattools.adfuller(train_log_diff1_12)


    st.write(f"La P-value de l'ADF est : {p_value:.10e}")
    if p_value < 0.05:
       st.write("(Inférieure à 0.05 : la série est stationnaire)")
    else:
       st.write("(Supérieure à 0.05 : la série n'est pas stationnaire)")


    st.subheader("Identification des Paramètres du Modèle (ACF et PACF)")
  
    st.write("""Les graphiques d'autocorrélation (ACF) et d'autocorrélation partielle (PACF) permettent d'identifier les paramètres ( p ), ( d ), ( q ) (pour ARIMA) ainsi que les paramètres saisonniers ( P ), ( D ), ( Q ) pour SARIMA.""")

    st.write("La décroissance progressive de l'ACF et les pics dans le PACF suggèrent l'utilisation d'un modèle SARIMA (1,1,1)(0,1,1)[12].")

    st.pyplot(fig_acf)
    st.pyplot(fig_pacf)
    st.write("""Après analyse des composants de tendance et de saisonnalité, ainsi que des résultats des tests de stationnarité et des graphiques ACF/PACF, nous optons pour un modèle **SARIMA (1,1,1)(0,1,1)[12]**.
             Ce modèle est bien adapté pour capturer les dynamiques saisonnières mensuelles observées dans les données, en particulier pour les pics hivernaux et les creux estivaux.""")
    st.write("Exemple : ")

    st.subheader("Model SARIMAX")
    st.write(loaded_results.summary())
    st.write("Les pvalues sont pour l'essentiel inférieur à 0.05.Le test de Ljung-Box est un test de blancheur des résidus. C'est un test statistique qui vise à rejeter ou non l'hypothèse 𝐻0 : Le résidu est un bruit blanc. Ici on lit sur la ligne Prob(Q) que la p-valeur de ce test est de 0.90 , donc on ne rejette pas l'hypothèse.Le test de Jarque-Bera est un test de normalité. C'est un test statistique qui vise à rejeter ou non l'hypothèse 𝐻0 : Le résidu suit une distribution normale. Ici on lit sur la ligne Prob (JB) que la p-valeur du test est de 0.00 . On rejette donc l'hypothèse.Les résidus confirme les hypothéses que nous avons faite initialement, ce qui conforte notre modèle.")
 
    st.subheader("Métriques SARIMAX sur 24 Mois")
    st.write ("Le score est:" ,score)
    st.write("Les métriques sont les suivantes :")
    st.dataframe(data)
    st.write("Les résultats nous montrent ici une bonne capacité du modèle à prédire la consommation énergétique, nous avons un score très proche de 1 et des MAE, MSE, RMSE tout a fait acceptable.")


    st.pyplot(sarima)
    st.write("Ce graphique confirme que le modèle SARIMAX est bien adapté à notre série temporelle, offrant une base solide pour des décisions futures sur la gestion énergétique.")

    st.subheader("Entrainement du modèle sur des données plus récente")

    st.write("Nous allons exéctuer notre model sur des données allant de 2013 jusqu'en Avril 2024") 

    st.subheader("Model SARIMAX suite au nouvel entrainement de donnée ")
    st.write ("Le résultat du model est le suivant :")
    st.write(resultat2)

    st.subheader("Métriques SARIMAX sur 12 Mois")
    st.write ("Le score est:" ,score2)
    st.write("Les métriques sont les suivantes :")
    st.dataframe(data2)
    st.write("A travers ces métriques, nous pouvons constater que notre modèle s'adapte bien aux nouvelles données.")
   
 
    st.subheader("Comparaison des prédictions initiales avec les données réel et estimation future")
    st.pyplot(sarima2)
    st.write("Ce graphique confirme que notre estimation intiale (faite via les données intialies), est bien cohérente avec les données réelles. Notre modèle semble bien adapté aux données de consommation électrique. Il capte bien les variations, et donne des estimations cohérentes.")

    st.write("Les métriques entre les données estimées (via le jeu de donnée intiale) et les données réelles sont :")
    st.write("Le score :",score2_1)
    st.dataframe(data2_1)
    st.write("Les métriques comparent les données réels aux estimations réalisées avec le jeu de donnée initiale. Les résultats obtenus nous permette de confirmer le modèle.")

  elif model_choisi=="Comparaison SARIMAX/PROPHET" :
  
    st.subheader("Comparaison des 2 modèles")
    
 
    st.write('Les métriques sur 12 mois de prophet pour ' ,selection_region ,'sont :',perf)
    st.write('Les métriques de SARIMAX pour', selection_region,'sur 12 mois sont :')
 
    st.dataframe(data_12)

    st.write("""Nous pouvons constater que les métriques du modèle SARIMAX sont meilleures à l'horizon 12 mois que le modèle Prophet.
     Cependant, comme nous pouvons le voir dans la modélisation Prophet, plus l'horizon est proche, plus les métriques Prophet s'améliorent. 
     Nous allons donc privilégier SARIMA pour des prévisions sur le moyen, long terme et Prophet sur le court terme.""")
   
    comparaison_idf= model_prophet.plot(forecast)
    plt.plot(y_pred_sarima,label='prediction SARIMA',color='green')
    plt.legend()
    plt.title("Prediction SARIMA vs PROPHET")

  
    comparaison_bfc= model_prophet.plot(forecast)
    plt.plot(y_pred_sarima_BFC,label='prediction SARIMA',color='green')
    plt.legend()
    plt.title("Prediction SARIMA vs PROPHET")



    if selection_region=="Île-de-France" :  
        st.pyplot(comparaison_idf)
    elif selection_region=='Bourgogne-Franche-Comté' :
         st.pyplot(comparaison_bfc)

  

  
    st.write ("""
    Le graphique compare les prédictions des deux modèles et confirme notre précédent comparatif :    
    - SARIMA capture très bien les cycles saisonniers avec des prédictions régulières en étant très bon sur des horizons à long terme.
    - Prophet suit également les tendances mais semble plus efficace à court terme.
    """)


if page==pages[4] :
    
 st.image("Energie_2.jpg")    

 st.write("### Synthèse")

 st.write("Pour résumer, dans le cadre de notre projet de fin de formation en analyse de données, cette étude approfondie sur l'estimation de la consommation électrique future en France nous a permis de comprendre les tendances de consommation afin de contribuer à assurer une transition énergétique réussie. Notre travail s'est appuyé sur la collecte et l'analyse de divers jeux de données, notamment les données éCO2mix régionales sur la consommation et la production électrique par source d'énergie, les données météorologiques (températures, rayonnement solaire, vitesse du vent) et des informations sur les secteurs d'activité liés à la consommation électrique. Un effort significatif a été consacré au nettoyage et à la préparation de ces données pour garantir la fiabilité de nos analyses. Des tests statistiques, tels que les coefficients de corrélation de Pearson, ont permis d'identifier les variables significatives influençant la consommation électrique. L'analyse a révélé une saisonnalité marquée dans la consommation électrique, avec des pics en hiver et des creux en été. Pour modéliser ces tendances, nous avons utilisé des modèles de séries temporelles, notamment le modèle SARIMA, adapté pour capturer la saisonnalité et les tendances à long terme. Les prédictions sur 12 mois ont montré une excellente adéquation avec les données réelles, avec un score de 0,94. Parallèlement, nous avons employé la librairie Prophet, qui permet d'intégrer des variables externes comme la température pour affiner les prévisions. Prophet a démontré une meilleure performance sur les prévisions à court terme. Le comparatif des deux modèles a mis en évidence que SARIMA est plus robuste pour les prévisions à long terme, tandis que Prophet offre une précision accrue à court terme.")

 st.write("Notre étude a mis en évidence une tendance générale à la baisse de la consommation électrique en France. Cette diminution, confirmée par nos analyses statistiques et nos modèles de prévision, s'inscrit dans un contexte de transition énergétique et de prise de conscience accumulée des enjeux environnementaux. Plusieurs facteurs expliquent cette tendance : l'augmentation des tarifs d'électricité incite les consommateurs à réduire leur consommation ; les technologies avancées ont conduit à l'adoption d'équipements plus efficaces et moins énergivores ; le changement climatique entraîne des hivers plus doux, notamment la demande en chauffage électrique ; une sensibilisation environnementale croissante conduit la population à adopter des comportements plus responsables ; enfin, les politiques gouvernementales mettent en place des mesures incitatives pour réduire la consommation et promouvoir les énergies renouvelables.")

 st.write("Les implications de cette baisse sont significatives. Elle facilite l'atteinte des objectifs de décarbonation fixés lors d'accords internationaux tels que la COP22, en particulier les émissions de gaz à effet de serre liées à la production d'électricité. De plus, elle diminue la pression sur le réseau électrique national, améliorant sa résilience face aux variations de la demande et aux aléas climatiques. Toutefois, cette tendance pose également des défis aux producteurs d'énergie, qui doivent adapter leurs modèles économiques et investir dans les énergies renouvelables et les technologies de stockage.")

 st.write("Bien que notre étude ait apporté des éclairages significatifs sur l'évolution de la consommation électrique, plusieurs pistes méritent d'être explorées pour approfondir cette analyse. Nous aurions aimé affiner notre modélisation en travaillant sur des données hebdomadaires, voire journalières, notamment en utilisant la librairie Prophet, ce qui permet de capturer les fluctuations à court terme et d'identifier des schémas de consommation plus précis. Une analyse à une échelle plus locale, par exemple au niveau des villes, offrirait une compréhension détaillée des variations régionales. De même, étudier l'impact de la densité de population ou distinguer entre les villes industrielles et tertiaires permettra d'évaluer l'influence des activités économiques sur la consommation énergétique.")

 st.write("En somme, ces perspectives ouvrent la voie à une compréhension plus fine et nuancée de la consommation électrique en France. Elles soulignent l'importance d'une approche multidimensionnelle, combinant analyses techniques approfondies et prise en compte des facteurs socio-économiques. Poursuivre dans cette direction contribuera non seulement à optimiser la gestion énergétique, mais aussi à soutenir les efforts de transition vers une société plus durable et résiliente face aux défis climatiques.")

 st.subheader("Annexe") 
 st.write("""
 Les sources des données sont :
 - Consommation quotidienne brute régionale  : https://odre.opendatasoft.com/explore/dataset/consommation-quotidienne-brute-regionale/export/?disjunctive.code_insee_region&disjunctive.region&refine.region=%C3%8Ele-de-France&refine.region=Bourgogne-Franche-Comt%C3%A9")
 - Données éCO2mix régionales consolidées et définitives (janvier 2013 à janvier 2023) : https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/information/?disjunctive.libelle_region&disjunctive.nature&sort=-date_heure
 - Température quotidienne régionale (depuis janvier 2016) : https://www.data.gouv.fr/fr/datasets/temperature-quotidienne-regionale-depuis-janvier-2016/#_
 - Consommation et thermosensibilité d'électricité annuelles à la maille région (de 2011 à 2023) : https://data.enedis.fr/explore/dataset/consommation-electrique-par-secteur-dactivite-region/table/?sort=annee
 """)
