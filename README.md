# Generative Jazz Musik
* GDKI WiSe 22/23
* Gruppe A:
    * Alexandra Schnakenberg - 
    * Fabien Manger - 
    * Peer Schliephacke - 5180417

## Kurzbeschreibung
Im Rahmen der Prüfungsleistung für das Modul 'Grundlagen der Künstlichen Intelligenz' stellt dieses Projekt ein rekurrentes Neuronales Netz dar, welches genutzt wird um generative Jazz Musik zu schaffen.
Die generative Jazz Musik wird dabei als sogenanntes Jazz Lick generiert. Ein Jazz Lick beschreibt ferner eine kurze Phrasologie über ein kadenzielles Harmonie Schema. Derartige Jazz Licks werden somit für improvisationen verwendet und können in vielfältiger Form auftauchen. Im Rahmen dieses Projekts werden drei taktige II-V-I Jazz Licks in Dur generiert.
Das Projekt geht damit der zentralen Fragestellung nach, ob ein Neuronales Netz in der Lage ist neue, akzeptabel klingende Jazz Licks zu generieren. Diese werden in diversen Analysen/Metriken validiert.

## Installation
Zur Verwendung des Repositorys wird eine funktionierende jupyter Umgebung vorrausgesetzt. 
Ferner sind diverse Python Module notwendig, die mithilfe der 'requirements.txt' und pip installiert werden können:
```
pip install -r requirements.txt
```
Zu dem ist eine aktuelle Version der Software 'Musescore3' notewendig, um die Midi-Dateien abzuspielen (https://musescore.com/)

## Ausführung
* Das Projekt wird sequenziell über 3 Verschiedene Notebooks ausgeführt
* Insgesamt werden 3 verschiedene Modelle trainiert, die auf verschiedenen Jazz Skalen und Tonmaterial basieren (Diatonisch, alteriert und beides zusammen) - daraus folgt die 3-fache Anzahl an gespeicherten Daten und Vorgängen.
* Die Notebooks erfüllen folgenden Zweck:
    * 1_LSTM_generated_weights: Dient zum Einlesen und Prozessieren der Trainingsdaten (Midi-format) in ein adäquates Format, das vom Netzwerk eingelesen werden kann. Letztlich werden die Netzwerke mit den Trainingsdaten trainiert und die Gewichte gespeichert. Der Trainingsvorgang umfasst diverse Hyperparameter die nach belieben gesetzt werden können.
    * 2_LSTM_generate_lick: Dient zum generieren von neuen Jazz Licks. Dabei werden die zuvor trainierten Gewichte genutzt um darauf basierend Vorhersagen zu machen. Analog existieren diverse Hyperparameter mit denen der Generierungs bzw Vorhersageablauf reguliert werden kann.
    * 3_Modell_Evaluation_and_Validation: Dient lediglich zum Vergleich der generierten Jazz Licks mit den Trainingsdaten. Ziel ist es die Verallgemeinerungsfähigkeit sowie die Mustererkennung der Netzwerk-Modelle bewerten zu können. Dabei werden neben einem Turing-Test ähnlichen Ansatz statistische sowie Musikwissenschaftliche Analysen getätigt.

## Datei-Übersicht
* Folgende Dateien und Ordner gehören zu dem Projekt:
Ordner:
    * Sample_Audios: Beinhaltet die Audio Dateien die für den Turing-Test ähnlichen Ansatz genommen werden. Die Audio Dateien werden mit Musescore erstellt (aus den Midi Dateien) und mit einer adäquaten Harmonischen Begleitung sowie einer ternären Interpretation des Rhytmus (Swing) ergänzt. Die Audio Dateien werden zufällig gezogen (10 insgesamt = 5 Originale Licks + 5 generierte Licks)
    * data: Beinhaltet alle digitalisierten Trainingsdaten - gefiltert in alteriert und diatonisch
    * generated_midi: Beinhaltet alle generierten Licks (aus 2_LSTM_generate_lick) unterteilt in Ordnern mit der Epochenanzahl. Alle neu generierten Licks werden in dem Ordner mit der jeweiligen Skala gespeichert (alteriert, diatonisch oder beide)
    * imgs: Beinhaltet Bilder über die Architektur oder aus der Evaluierung/Validierung
    * stored: Beinhaltet Informationen im Binärformat (aus 1_LSTM_generated_weights) um effizient Daten zwischen den Notebooks zu übertragen. Die Binär-Dateien werden Skalenspezifisch den jeweiligen Ordnern zugeordnet.
    * weights: Speichert Checkpoints (wenn Optionaler Parameter gesetzt ist) und die trainierten Gewichte aus 1_LSTM_generated_weights um auf Grundlage dessen Licks zu generieren (in 2_LSTM_generate_lick)
    * documents: Beinhaltet die Dokumentation, PowerPointPräsentation und das Poster im pdf Format
* utils enthält ausgelagerten Python code der durch implementierte Funktionen zur Bearbeitung und Programmieren der eigentlichen Ziele beschrieben wird. Durch die ausgelagerten Funktionen werden Schnittstellen gebildet und die Ordnung der Notebooks aufrecht gehalten:
    * evaluate.py : Beinhaltet Methoden zum generieren von Grafiken für die Validierung
    * jazz_lstm.py : Beinhaltet die Netzwerkarchitektur
    * midi_generation.py : Beinhaltet Funktionen zur Generierung der neuen Jazz Licks im Midi Format
    * midi_tools.py : Beinhaltet Funktionen zum Einladen und Transformieren der Trainingsdaten
    * check_overfitting.py : Überprüft ob ein generiertes Jazz-Lick einfach aus den Trainingsdaten kopiert wurde, in dem jede Notensequenz aus dem Ordner mit jeder Notensequenz aus den Trainingsdaten verglichen wird (Rhythmus wird ignoriert). Dabei werden durch geeigente Visualisierungen der Anteil an überangepassten Licks aufgezeigt sowie die Namen der validen Licks angezeigt (not overfitted).
* Notebooks: Siehe Ausführung

## Notiz
* Die Fertigen Audio Dateien werden mit einer maginalen musescore-generierten Begleitung (cm9-F7-Bbmaj9) und der Jazztypischen Rhytmus-Interpretation, dem Swing, ergänzt um die Licks in einer realistischen Klangumgebung einzubetten.  
* Die Python Dateien wurden ausführlich dokumentiert und jeweils mit einem Docstring ergänzt sodass innerhalb des Notebooks schnell eine Erklärung angezeigt werden kann
* Die Notebooks enthalten zusätzliche Informationen über das Ziel der jeweiligen Notebooks
* Die Dokumentationssprache der Notebooks/Python dateien ist englisch, da im Sinne vieler Konventionen englisch als Variablennamen genutzt wird und damit der Python code sprachlich konsitent sein soll
# GJL
