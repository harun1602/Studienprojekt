from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text, Table, Double, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

Base = declarative_base()

# Many-to-Many Beziehung zwischen TaskProfile und Version über die sekundäre Tabelle
# Verbindungstabelle: TaskProfile <-> Version (Ein Aufgabenprofil kann mehrere Versionen beinhalten und eine Version kann in mehreren Profilen vorkommen)
task_profile_version_association = Table(
    'task_profile_version', Base.metadata,
    Column('task_profile_id', Integer, ForeignKey('task_profiles.id')),
    Column('version_id', Integer, ForeignKey('versions.id'))
)

# Many-to-Many-Beziehung für User zu taskProfilen
# Verbindungstabelle: User <-> TaskProfile
user_taskprofile_association = Table(
    'user_taskprofile', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('task_profile_id', Integer, ForeignKey('task_profiles.id'))
)

# Many-to-Many-Beziehung für ComponentList zu Component
# Eine Bauteilliste enthält mehrere Komponenten, eine Komponente kann in mehreren Bauteillisten vorkommen
component_list_components_association = Table(
    'component_list_components', Base.metadata,
    Column('component_list_id', Integer, ForeignKey('component_lists.id')),
    Column('component_id', Integer, ForeignKey('components.id'))
)


# Allgemeine Settings -> Persistent Speichern
class Setting(Base):
    """Tabelle zur Speicherung für die allgemeinen Einstellungen im Admin-Task Management."""
    __tablename__ = 'settings'

    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True)
    value = Column(String)


# Definition User-Tabelle
class User(Base):
    """
    Tabelle für Benutzer mit allgemeinen Attributen.

    Relationships:

    tasks: Aufgaben, die der Nutzer durchgeführt hat
    task_profiles: Aufgabenprofile denen der Nutzer zugewiesen wurde
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    firstname = Column(String, nullable=False)
    lastname = Column(String, nullable=False)
    nickname = Column(String, nullable=False, unique=True)
    age = Column(Integer, nullable=False)
    skill = Column(String, nullable=False)

    tasks = relationship('Task', back_populates='user')
    task_profiles = relationship('TaskProfile', secondary=user_taskprofile_association, back_populates='users')


# Definition Tabelle für einzelnen Aufgaben-Schritte
class TaskStep(Base):
    """
    Tabelle für einzelnen Aufgaben-Schritte.

    Relationships:

    task: Verweis auf die zugeordnete Aufgabe
    """
    __tablename__ = 'task_steps'

    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey('tasks.id'))
    step_number = Column(Integer, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    time_spent = Column(Integer, nullable=True)

    task = relationship('Task', back_populates='task_steps')


# Definition Tabelle Gesamtaufgabe
class Task(Base):
    """
    Tabelle für Gesamtaufgaben

    Relationships:

    user: Verweis auf den Nutzer, der die Aufgabe gelöst hat
    version: Verweis auf Aufgaben-Version
    task_steps: Aufgabenschritte der Aufgabe
    version_history: Verweis auf Historie
    """
    __tablename__ = 'tasks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_id = Column(Integer, ForeignKey('versions.id'))
    version_history_id = Column(Integer, ForeignKey('version_histories.id'))  # Verknüpfung zur VersionHistory
    user_id = Column(Integer, ForeignKey('users.id'))
    time = Column(Integer, nullable=False)
    start_timestamp = Column(DateTime, default=datetime.now)
    end_timestamp = Column(DateTime, nullable=True)
    perceived_complexity = Column(Integer, nullable=True)
    perceived_stress = Column(Integer, nullable=True)
    perceived_time_pressure = Column(Integer, nullable=True)
    perceived_frustration = Column(Integer, nullable=True)
    game_mode = Column(Integer, nullable=True)

    user = relationship('User', back_populates='tasks')
    version = relationship('Version', back_populates='tasks')
    task_steps = relationship('TaskStep', back_populates='task', cascade='all, delete-orphan')
    version_history = relationship('VersionHistory', back_populates='tasks')


# Definition Tabelle für Versionen
class Version(Base):
    """
    Tabelle für Versionen

    Relationships:

    images: Verweis auf die Anleitungsbilder
    tasks: Aufgaben
    version_history: Verweis auf Historie
    """
    __tablename__ = 'versions'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    complexity = Column(Double, nullable=False)
    time_limit = Column(Integer, nullable=False)
    cover_image_path = Column(String, nullable=True)
    images = relationship("Image", back_populates="version")
    tasks = relationship('Task', back_populates='version')
    active = Column(Boolean, default=True)

    history = relationship("VersionHistory", back_populates="version", cascade="all, delete-orphan")


# Definition Tabelle für Versionshistorie (alte Versionen speichern)
class VersionHistory(Base):
    """
    Tabelle für Versionshistorie

    Relationships:

    version: Verweis auf Aufgaben-Version
    image_histories: Verweis auf die historischen Anleitungsbilder
    tasks: Verweis auf Tasks, die auf gegebener Revision beruhen
    """
    __tablename__ = 'version_histories'
    id = Column(Integer, primary_key=True, autoincrement=True)
    version_id = Column(Integer, ForeignKey('versions.id'))
    revision = Column(Integer, nullable=False)  # Revisionsnummer
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    complexity = Column(Double, nullable=False)
    time_limit = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

    version = relationship("Version", back_populates="history")
    image_histories = relationship("ImageHistory", back_populates="version_history", cascade='all, delete-orphan')
    tasks = relationship("Task", back_populates="version_history")


# Definition Tabelle für Bilder in den Versionen
class Image(Base):
    """
    Tabelle für Bilder in den Versionen

    Relationships:

    version: Verweis auf Aufgaben-Version, die dem Bild zugeordnet sind
    history: Verweis auf die historischen Anleitungsbilder
    """
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    version_id = Column(Integer, ForeignKey('versions.id'))
    image_path = Column(String, nullable=False)
    image_anleitung = Column(String, nullable=False)

    version = relationship("Version", back_populates="images")
    history = relationship("ImageHistory", back_populates="image", cascade='all, delete-orphan')
    required_components = relationship('TaskComponentRequirement', back_populates='image', cascade="all, delete")


# Definition Tabelle für Bilderhistorie
class ImageHistory(Base):
    """
    Tabelle für Bilderhistorie
    """
    __tablename__ = 'image_histories'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    version_history_id = Column(Integer, ForeignKey('version_histories.id'))  # Verknüpfung mit der VersionHistory
    revision = Column(Integer, nullable=False)  # Revisionsnummer
    image_path = Column(String, nullable=False)
    image_anleitung = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

    image = relationship("Image", back_populates="history")
    version_history = relationship("VersionHistory", back_populates="image_histories")


# Definition Tabelle der Aufgabenprofile
class TaskProfile(Base):
    """
    Tabelle der Aufgabenprofile, mehrere Versionen zu einem Profil

    Relationships:

    versions: Verweis auf Aufgaben-Versionen des Profils
    required_counts: Wie häufig Versionen durchgeführt werden soll
    users: Nutzer, die diesem Profil zugeordnet wurden
    """
    __tablename__ = 'task_profiles'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    versions = relationship('Version', secondary=task_profile_version_association, backref='task_profiles')
    required_counts = relationship('TaskProfileRequiredCount', back_populates='task_profile')
    users = relationship('User', secondary=user_taskprofile_association, back_populates='task_profiles')
    active = Column(Boolean, default=True)


# Definition Tabelle für benötigte Anzahl
class TaskProfileRequiredCount(Base):
    """
    Tabelle für benötigte Anzahl einer Version in einem TaskProfile

    Relationships:

    version: Verweis auf Aufgaben-Version
    task_profile: Verweis auf das Aufgabenprofil
    """
    __tablename__ = 'task_profile_required_counts'

    id = Column(Integer, primary_key=True)
    task_profile_id = Column(Integer, ForeignKey('task_profiles.id'))
    version_id = Column(Integer, ForeignKey('versions.id'))
    count = Column(Integer, nullable=False)

    task_profile = relationship('TaskProfile', back_populates='required_counts')
    version = relationship('Version')


# Definition Tabelle für Components(Bauteile) -> Beispielsweise Schraube
class Component(Base):
    """
    Tabelle für einzelne Bauteile und Bilder -> Beispielsweise Schraube
    """
    __tablename__ = 'components'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    component_image_path = Column(String, nullable=True)


# Definition Tabelle ComponentList(Bauteilliste)
class ComponentList(Base):
    """
    Tabelle für die Bauteilliste einer Version

    Relationships:

    components: Alle Komponenten einer Liste
    required_counts: Anzahl der benötigten Komponenten
    """
    __tablename__ = 'component_lists'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    version_id = Column(Integer, ForeignKey('versions.id'))

    components = relationship('Component', secondary=component_list_components_association, backref='component_lists')
    required_counts = relationship('ComponentListRequiredCount', back_populates='component_list', cascade="all, delete")


# Definition Tabelle zum Speichern der benötigten Anzahl
class ComponentListRequiredCount(Base):
    """
    Tabelle für benötigte Anzahl eines Bauteils in einer Liste
    """
    __tablename__ = 'component_list_required_counts'

    id = Column(Integer, primary_key=True)
    component_list_id = Column(Integer, ForeignKey('component_lists.id'))
    component_id = Column(Integer, ForeignKey('components.id'))
    count = Column(Integer, nullable=False)

    component_list = relationship('ComponentList', back_populates='required_counts')
    component = relationship('Component', cascade="all, delete")


class TaskComponentRequirement(Base):
    __tablename__ = 'task_component_requirements'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    component_id = Column(Integer, ForeignKey('components.id'))
    count = Column(Integer, nullable=False)

    image = relationship('Image', back_populates='required_components')
    component = relationship('Component')



def create_default_users(session):
    """Erstellt alle benötigten Default-Nutzer"""
    admin_user = session.query(User).filter_by(firstname='ADMIN', lastname='ADMIN').first()
    if not admin_user:
        admin_user = User(firstname='ADMIN', lastname='ADMIN', nickname='ADMIN', age=0, skill='ADMIN')
        session.add(admin_user)
        session.commit()

    ben_user = session.query(User).filter_by(firstname='Ben', lastname='Olschar').first()
    if not ben_user:
        admin_user = User(firstname='Ben', lastname='Olschar', nickname='Benno', age=21, skill='Informatik', task_profiles=[session.query(TaskProfile).first()])
        session.add(admin_user)
        session.commit()


def create_defaults(session):
    """Erstellen der Standardversion, falls diese nicht existiert"""

    default_version_name = 'default_version'
    default_version = session.query(Version).filter_by(name=default_version_name).first()
    if not default_version:

        # Definiere Standardbilder
        default_images = [
            {
                'image_path': f'images/{default_version_name}/revision_1/default_image1.jpg',
                'image_anleitung': 'Anleitung für das default_image1'
            },
            {
                'image_path': f'images/{default_version_name}/revision_1/default_image2.jpeg',
                'image_anleitung': 'Anleitung für das default_image2'
            }
        ]

        name = default_version_name
        description = 'Es handelt sich um eine default Beschreibung'
        complexity = 1
        cover_image_path = f'images/{default_version_name}/cover_Cover.jpeg'
        time_limit = 100
        images = default_images

        # Neue Version anlegen
        default_version = Version(
            name=name,
            description=description,
            complexity=complexity,
            cover_image_path=cover_image_path,
            time_limit=time_limit
        )

        session.add(default_version)
        session.commit()

        # Erste Revision für die Version
        new_version_history = VersionHistory(
            version_id=default_version.id,
            revision=1,
            name=name,
            description=description,
            complexity=complexity,
            time_limit=time_limit
        )
        session.add(new_version_history)
        session.commit()

        # Bilder für die neue Version
        for image in images:
            new_image = Image(
                version_id=default_version.id,
                image_path=image['image_path'],
                image_anleitung=image['image_anleitung']
            )
            session.add(new_image)
            session.commit()

            # History für jedes Bild
            new_image_history = ImageHistory(
                image_id=new_image.id,
                version_history_id=new_version_history.id,  # Verknüpfung zur VersionHistory
                revision=1,
                image_path=image['image_path'],
                image_anleitung=image['image_anleitung']
            )
            session.add(new_image_history)

        session.commit()


    # Erstellen des Standardarbeitsprofils, falls dieses nicht existiert
    default_task_profile_name = 'default_task_profile'
    default_task_profile = session.query(TaskProfile).filter_by(name=default_task_profile_name).first()
    if not default_task_profile:
        default_task_profile = TaskProfile(name=default_task_profile_name)
        default_task_profile.versions.append(default_version)
        session.add(default_task_profile)
        session.commit()
        required_count = TaskProfileRequiredCount(task_profile_id=default_task_profile.id, version_id=default_version.id, count=3)
        session.add(required_count)
        session.commit()



engine = create_engine('sqlite:///ProSchedulerDatabaseAlembic.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


# Erstelle defaults
create_defaults(session)

# Admin-User erstellen, falls nicht vorhanden
create_default_users(session)
