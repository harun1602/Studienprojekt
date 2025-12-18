import pandas as pd


def average_task_completion_time_per_version_panda(tasks_df: pd.DataFrame, versions_df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet die durchschnittliche Bearbeitungszeit pro Version"""
    # Führe Tasks- und Versionsdaten zusammen
    tasks_with_versions = pd.merge(tasks_df, versions_df, left_on='version_id', right_on='id', suffixes=('', '_version'))

    # Berechne den Durchschnitt der Abschlusszeiten pro Version
    avg_completion_time_per_version = tasks_with_versions.groupby('name')['time'].mean().reset_index()
    avg_completion_time_per_version.columns = ['Version', 'Durchschnittliche Zeit']

    return avg_completion_time_per_version


def std_dev_task_completion_time_per_version_panda(tasks_df: pd.DataFrame, versions_df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet die Standardabweichung der Bearbeitungszeiten pro Version"""
    # Führe Tasks- und Versionsdaten zusammen
    tasks_with_versions = pd.merge(tasks_df, versions_df, left_on='version_id', right_on='id', suffixes=('', '_version'))

    # Berechne die Standardabweichung der Abschlusszeiten pro Version
    std_dev_completion_time_per_version = tasks_with_versions.groupby('name')['time'].std().reset_index()
    std_dev_completion_time_per_version.columns = ['Version', 'Standard Abweichung']

    return std_dev_completion_time_per_version


def average_perceived_complexity_per_version_for_user(tasks_df: pd.DataFrame, versions_df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """Berechnet die durchschnittliche wahrgenommene Komplexität je Version für einen bestimmten Nutzer"""
    # Filter für den bestimmten Nutzer
    user_tasks = tasks_df[tasks_df['user_id'] == user_id]

    # Füge die Versionen-Daten an die User-Tasks-Daten an
    user_tasks_with_versions = pd.merge(user_tasks, versions_df, left_on='version_id', right_on='id', suffixes=('', '_version'))

    # Berechne die durchschnittliche wahrgenommene Komplexität pro Version
    avg_complexity_per_version = user_tasks_with_versions.groupby('name').agg(
        Durchschnittlich_wahrgenommene_Complexity=('perceived_complexity', 'mean'),
        Tatsächliche_Complexity=('complexity', 'first')
    ).reset_index()

    avg_complexity_per_version.columns = ['Version', 'Durchschnittlich wahrgenommene Complexity', 'Tatsächliche Complexity']

    return avg_complexity_per_version


def average_perceived_complexity_per_version_for_all(tasks_df: pd.DataFrame, versions_df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet die durchschnittliche wahrgenommene Komplexität je Version über alle Nutzer hinweg"""
    # Füge die Versionen-Daten an die User-Tasks-Daten an
    user_tasks_with_versions = pd.merge(tasks_df, versions_df, left_on='version_id', right_on='id', suffixes=('', '_version'))

    # Berechne die durchschnittliche wahrgenommene Komplexität pro Version
    avg_complexity_per_version = user_tasks_with_versions.groupby('name').agg(
        Durchschnittlich_wahrgenommene_Complexity=('perceived_complexity', 'mean'),
        Tatsächliche_Complexity=('complexity', 'first')
    ).reset_index()

    avg_complexity_per_version.columns = ['Version', 'Durchschnittlich wahrgenommene Complexity', 'Tatsächliche Complexity']

    return avg_complexity_per_version


def filter_default_version(df: pd.DataFrame):
    if 'Version' not in df.columns:
        return df

    # Prüfen, ob es andere Versionen als default_version gibt
    if df['Version'].nunique() > 1:

        # Zeilen mit 'default_version' entfernen
        df_filtered = df[df['Version'] != 'default_version']
        return df_filtered
    else:
        # Falls es keine anderen Versionen gibt, das ursprüngliche DataFrame zurückgeben
        return df