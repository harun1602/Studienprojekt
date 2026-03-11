# DemonstratorProzesszeitprognose

**Beschreibung:**  
Dieses Repository enthält einen Demonstrator zur Erweiterung eines bestehenden Werkassistenzsystems (WAS) der LPS Lern- und Forschungsfabrik (LFF). Das System unterstützt Montagearbeitende bei der Durchführung einzelner Montageschritte und stellt prozessrelevante Informationen über eine Benutzeroberfläche bereit.

Der Demonstrator integriert eine kamerabasierte Bilderkennung zur Analyse des Montageprozesses. Mithilfe von Computer-Vision-Verfahren können Montagezustände erkannt und potenzielle Fehler oder Abweichungen während der Ausführung eines Arbeitsschritts identifiziert werden. Die erkannten Informationen werden im User Interface dargestellt und können von Mitarbeitenden direkt zur Fehlerkorrektur genutzt werden.

Zusätzlich werden Montagezeiten erfasst und gespeichert, um eine Prozesszeitprognose zu ermöglichen. Auf Basis der aufgenommenen Daten können Montagezeiten für neue Produktvarianten und individuelle Nutzende geschätzt werden.


## Voraussetzungen
> [!IMPORTANT]
> Dieses Projekt erfordert Python **3.12**. Bitte stelle sicher, dass du diese Version installiert hast, bevor du fortfährst. Ältere Versionen von Python werden nicht funktionieren und zu unerwarteten Fehlern führen.

### Installation von Python 3.12

#### Windows:
1. Besuche die offizielle Python-Website: [python.org/downloads](https://www.python.org/downloads/)
2. Lade den Installer für Python 3.12 herunter.
3. Während der Installation **"Add Python to PATH"** aktivieren.
4. Führe die Installation durch.

## Mit Projekt durchstarten

### Schritt 1: Repository klonen
Beginne damit, das Projekt-Repository auf deinem Computer zu klonen. Öffne dazu dein Terminal und führe folgenden Befehl aus:

```bash
git clone https://github.com/AIStudienprojektUnilokk/DemonstratorProzesszeitprognose.git
```

### Schritt 2: Virtuelle Umgebung erstellen
Wir empfehlen, eine virtuelle Umgebung zu verwenden, um die Abhängigkeiten des Projekts zu isolieren. Befolge dafür die folgenden Schritte:

1. **Wechsle in das Verzeichnis des Projekts:** <br> Der Name des Verzeichnisses ist i.d.R. der Name des Repositories:
```bash
cd DemonstratorProzesszeitprognose
git checkout v0.1.3-alpha
```

2. **Prüfe:** <br> Um sicher zu stellen, dass du im korrekten Verzeichnis bist, nutzte den ls Befehl, um dir den Inhalt des Verzeichnisses anzusehen:
```bash 
ls
```

3. **Erstelle eine virtuelle Umgebung:** <br> **Windows:** <br> 
```bash 
python -m venv venv
```
4. **Virutelle Umgebeung aktivieren** <br> **Windows:** 
```bash 
venv\Scripts\activate
``` 
### Schritt 3: Bibliotheken installieren
Die nötigen Bibliotheken und Frameworks für das Projekt sind in der Datei requirements.txt aufgelistet. Installiere diese mit folgendem Befehl:
```bash 
pip install -r requirements.txt
```

### Schritt 4: Projekt starten
Nun sollte dein Projekt erfolgreich aufgesetzt worden sein. Du kannst nun die Application mit folgendem Befehl starten:
```bash 
streamlit run streamlit_app.py
```

**Zugriff auf die Anwendung:** <br> Es sollte sich automatisch dein Webbrowser mit der URL http://localhost:8501. Wenn das nicht automatisch passiert ist, kannst du die URL manuell in deinen Browser eingeben.



