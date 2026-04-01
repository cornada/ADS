# Образовательные пространства и онтология обучения для HALA

## Executive Summary

HALA можно «приземлить» на исследовательски валидируемую модель *learning space* как **систему**, в которой обучение возникает из взаимодействия акторов, артефактов, правил и границ — а не просто из “контекста” или “курса”. В практично применимой (не философской) литературе это лучше всего поддерживают три линии: (1) **e-learning стандарты и формальные модели** (IMS Learning Design, SCORM/xAPI/cmi5, Caliper, CASE, IEEE LOM/ISO MLR) как языки описания структуры/трасс/результатов, citeturn6search15turn5search23turn12search2turn13search9turn5search1turn11search4turn12search7turn0search12; (2) **Cultural-Historical Activity Theory (CHAT)** и **expansive learning** как «онтология» образовательной системы (subject–object–tools–rules–community–division of labor) и как механизм переходов/трансформаций через противоречия, citeturn14search9turn14search0turn14search1turn14search26; (3) **learning sciences**-рамки, которые концептуализируют пространство как движение и участие: communities of practice/legitimate peripheral participation, knowledge building community space, acquisition vs participation metaphors, boundary crossing/boundary objects, citeturn3search0turn16search1turn15search1turn15search3turn1search17turn10search22.

Для HALA ключевой вывод: существующие образовательные онтологии и стандарты хорошо описывают **контент/активности/компетенции/трассы**, но слабо описывают **зоны** (exploration/practice/reflection/assessment), **протоколы переходов**, а также **психологическую безопасность, доверие и роль-границы** как *системные сущности*. Это открывает «белое пространство»: HALA может добавить слой **trust & safety governance** и слой **agent learning infrastructure** как часть ontology-first design, не ломая совместимость со стандартами (xAPI/Caliper/CASE). citeturn4search10turn4search3turn9search0turn12search2turn5search1turn11search4

Практическая рекомендация для экспериментальной реализации:  
- **Learning Design (план/сценарий)**: брать IMS Learning Design как «эталонно строгий» референс (пусть даже вы реализуете упрощённый DSL), потому что он формализует роли–активности–среды–метод, citeturn6search15turn6search1.  
- **Learning Traces (наблюдаемость процесса)**: описывать каждое действие агента и человека через xAPI statements (Actor–Verb–Object + Context/Result), а для LMS-подобного запуска/сессионности использовать cmi5 profile как «правила игры», citeturn12search2turn13search9turn13search16.  
- **Learning Analytics (интероперабельность евентов)**: там, где нужны общие словари событий, подключать Caliper, помня, что Caliper и xAPI «не одно и то же» и выбираются по use case, citeturn5search1turn6search6.  
- **Outcomes (компетенции/цели)**: описывать через CASE (skills/competencies/outcomes) и/или компетентностные онтологии (ELM/Core‑O/COMP2), citeturn11search4turn6search17turn6search36turn6search20.  
- **Assessment (встроенная оценка)**: проектировать по Evidence‑Centered Design (ECD) + (при необходимости) stealth assessment для «оценки внутри опыта», citeturn2search10turn2search3turn2search39turn9search0.  
- **Agent learning metadata (для AI‑агента)**: использовать ML‑Schema/MEX/OpenML‑стек как вдохновение/шаблон для описания «эксперимента обучения» (модель, данные, параметры, запуск, метрики, provenance) и связать это с HALA‑трассами, citeturn7search0turn8search13turn8search3turn8search0.

## Детальный анализ по восьми областям

**Онтология образовательных пространств (онтологии/стандарты/ядро сущностей).**  
В большинстве практических «онтологий образования» и стандартов повторяются одни и те же сущности, хотя часто они названы по‑разному: *Actor/Role* (ученик/преподаватель/система), *Activity/Event* (действие/взаимодействие), *Resource/Artifact* (learning object/материал/инструмент), *Context/Environment* (условия, платформа, группа), *Outcome/Competency* (ожидаемый результат), *Evidence/Result* (оценка/метрики/трассы), *Structure/Sequence* (модули/переходы), *Policy/Rules* (ограничения/права/правила). Это видно на уровне «планирования» (IMS Learning Design), «доставки» (SCORM/cmi5), «трассировки опыта» (xAPI/Caliper) и «описания результатов» (CASE, компетентностные онтологии). citeturn6search15turn5search23turn12search2turn5search1turn11search4  

Практически важная граница: **learning activities** часто описываются как события/процессы (xAPI statements, Caliper events), а **learning outcomes** — как стандартизированные компетенции/результаты (CASE, competence ontologies). Между ними обычно нет «жёсткой сцепки», и её приходится проектировать отдельно (через ECD или собственные mapping rules). citeturn12search2turn5search1turn11search4turn2search10  

Ещё одна практическая линия — Semantic Web стек (RDF/OWL) как язык формализации онтологий и связывания сущностей; FOAF здесь важен как пример словаря для описания людей/соц. связей, полезный как строительный блок «акторного слоя» learning space. citeturn12search1turn12search5turn12search0  

**Архитектура образовательных пространств (паттерны learning environments + зоны).**  
В дизайне learning environments (особенно в technology‑enhanced learning) устойчиво появляется идея, что пространство — это “range of places from real to virtual” и что дизайн пространства является агентом изменения практик. Это подчёркивается в традиции EDUCAUSE Learning Spaces. citeturn5search2turn5search14turn5search6  

Для HALA‑подобной постановки (AI‑агент учится под наблюдением человека) дизайн целесообразно делать **зонным**: exploration → guided practice → assessment gate → reflection/retrospective → consolidation/transfer. Исследовательская поддержка зонности приходит не как один стандарт, а как «сшивка» нескольких традиций:  
- *studio‑based learning* (циклы критики/ревизии) как модель практики и рефлексии (особенно в профессиональных дисциплинах), citeturn10search0turn10search12  
- *maker education / constructionism* как модель exploration‑and‑making с сильной ролью артефактов и публичности процесса, citeturn4search36turn4search5turn10search25turn4search24  
- *boundary objects* как мосты между зонами и сообществами практики (общие артефакты, которые допускают разные интерпретации, но сохраняют общую идентичность), citeturn10search22turn1search17.  

Для HALA особенно важны **transition protocols**: как система разрешает переходы между зонами (например, из exploration в practice, из practice в assessment). В образовательной литературе «механика переходов» часто изучается через boundary crossing: идентификация различий, координация, рефлексия, трансформация. Это можно «перевести» в протоколы HALA: check‑in/contract → shared artifact alignment → reflective review → schema update. citeturn1search17turn10search22  

**Теория Activity Systems (CHAT) как онтология образовательной системы.**  
CHAT/Activity Theory в версии Engeström задаёт очень прикладной «онтологический каркас» для образовательных пространств: *subject* (кто действует), *object* (на что направлена деятельность), *tools* (артефакты/инструменты), *rules* (нормы/ограничения), *community* (сообщество), *division of labor* (распределение ролей/ответственностей). Это прямо совпадает по форме с вашей гипотезой “actors–artifacts–processes–boundaries–rules”, но добавляет два критически важных элемента: (а) объект‑ориентированность деятельности (learning как производство/преобразование объекта), (б) противоречия как двигатель развития системы. citeturn14search9turn14search0turn14search26  

Expansive learning (как теория и как дизайн‑логика) полезна HALA именно потому, что описывает обучение как «конструирование того, чего ещё нет» через циклы преобразования практики, а не как линейное “освоение контента”. citeturn14search0turn14search9  

Практическая методология, особенно релевантная enterprise/партнёрствам человек–ИИ: Change Laboratory (formative intervention) — это «инструментированное пространство» для коллективной диагностики противоречий и перепроектирования деятельности. Для HALA это можно трактовать как референс‑паттерн для зоны reflection/re‑design (когда человек и агент вместе пересобирают правила, инструменты и интерфейсы). citeturn14search1turn14search5  

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["Engeström activity system triangle diagram","Change Laboratory activity theory method diagram","boundary crossing boundary objects education diagram","Knowledge Forum CSILE community knowledge space screenshot"],"num_per_query":1}

**Пространственные метафоры в образовании (landscape/territory/journey/network).**  
В learning sciences метафоры — не украшение, а способ выбрать «единицу анализа» и тип причинности. Классическая работа про это — различение *acquisition metaphor* и *participation metaphor*: либо мы думаем о знании как о «вещи, которую приобретают», либо о learning как о «вхождении в практику/участии». Для HALA это напрямую превращается в дизайн‑решение: агент “учится” как накоплением (acquisition: память/правила/skills) и как участием (participation: совместная деятельность, координация, ответственность). citeturn15search3turn15search30  

Communities of Practice и legitimate peripheral participation дают сильную модель «движения в пространстве практик»: новичок входит через периферийное участие и постепенно движется к более центральным ролям. Если перенести на HALA, то это почти буквальная проектная логика *scaffolded autonomy*: агент начинает как периферийный исполнитель под жёстким надзором и постепенно получает право на более центральные действия. citeturn3search0turn16search1  

Отдельно важна идея *landscapes of practice* и *knowledgeability* (умение ориентироваться и быть понятным на границах множества практик). Это можно трактовать как «метрику зрелости» агента A‑уровней: не только компетентность в одной задаче, но и способность **переходить между практиками/контекстами с минимальным разрушением доверия**. citeturn16search8turn16search3turn16search4  

Knowledge Building (Scardamalia & Bereiter) даёт очень прикладную трактовку «пространства знаний»: Knowledge Forum/CSILE как *community knowledge space* с «notes» и «views» — то есть пространство явно проектируется как общая среда для артефактов‑идей, их улучшения, связи и ревизий со временем. Для HALA это сильный референс для общего пространства «trace + rationale»: агент и человек должны совместно улучшать «объекты знания», сохраняя историю изменений. citeturn15search1turn15search8turn1search19  

Rhizomatic learning (community as curriculum) полезна как контраст: она расширяет идею пространства до «экосистемы», где учебные цели могут быть подвижными. Это поддерживает exploration‑зону HALA, но требует сильной инфраструктуры границ/безопасности, иначе система расползается. citeturn10search15turn10search11  

**Онтология для AI/Agent Learning (формальные модели “эксперимента обучения”).**  
В ML/agent мире уже существует зрелая практика описывать *эксперименты, пайплайны, данные, метрики и provenance* через онтологии/схемы (RDF/OWL‑совместимые): ML‑Schema как top‑level schema для datasets/algorithms/experiments, MEX vocabulary как lightweight описание ML‑экспериментов поверх PROV‑O, OntoDM как онтология data mining сущностей, OpenML как платформа, где эти идеи превращаются в инфраструктуру воспроизводимых “runs/tasks”. citeturn7search0turn8search13turn8search3turn7search1turn8search0  

Для HALA здесь главный integration point: **вы уже хотите learning space как наблюдаемую систему**. ML‑онтологии дают практические сущности, которых не хватает классическим edu‑онтологиям: *model version*, *hyperparameters*, *training run*, *evaluation metric*, *dataset lineage*, *environment configuration*, *provenance*. Их можно «свести» с образовательным слоем через общий концепт *trace/evidence*. citeturn8search11turn8search39turn8search3  

Потенциальная специфика agent learning (и пробел edu‑онтологий):  
- агент имеет **внутреннее состояние** (memory/policy) и **внешние инструменты** (tool APIs),  
- обучение может идти как обновление «policy» (переучивание), как обновление «memory» (постоянная память), и как обновление «правил поведения» (constitutional constraints),  
- критично важны *auditability* и *provenance* (кто изменил что и почему) — это ближе к ML metadata практикам, чем к LMS‑метаданным. citeturn7search0turn8search3turn12search2  

**Физические vs виртуальные образовательные пространства (presence/affordances/embodiment).**  
Практическая граница для HALA‑learning space: какие свойства физического пространства нужно “эмулировать” в цифровом? Литература про “learning spaces” показывает, что решающими являются не стены, а сочетание активностей, доступных инструментов и культурных норм взаимодействия (space as change agent). citeturn5search2turn5search14  

Концепт *affordances* помогает формализовать, что именно «позволяет» среда. В TEL‑литературе affordances связываются с тем, как обучающийся воспринимает и использует возможности технологии; это мост между “дизайн среды” и “действия в среде”. citeturn9search14turn9search34  

Для виртуальных пространств отдельно важна “presence” как ощущение «быть там». Современные обзоры подчёркивают, что presence нельзя сводить только к технике: это психологический феномен, зависящий от контекста, задач и когнитивных механизмов. citeturn9search7turn9search19  

Но есть и «охлаждающая» эмпирика: повышенная иммерсивность/embodiment в VR может повышать субъективные показатели presence и agency, но не гарантирует улучшения transfer (переноса знаний). Для HALA это предупреждение: “ощущение пространства” ≠ “результат обучения” — нужно проектировать evidence и feedback loops. citeturn9search27turn9search23  

**Assessment как часть пространства (embedded assessment + feedback loops).**  
Самое прикладное различение: *assessment for learning* (формирующая оценка, встроенная в процесс) vs *assessment of learning* (итоговая проверка). Классический обзор показывает, что частая качественная обратная связь и дизайн оценивания, встроенный в учебные практики, дают существенные эффекты. citeturn9search0turn9search4  

Evidence‑Centered Design (ECD) даёт “онтологию оценивания” в виде связки: *claims* (что утверждаем о компетентности), *evidence* (какие наблюдаемые признаки), *tasks* (какие задания провоцируют доказательства), *scoring/inference model* (как обновляем представление о способности). Это идеально подходит HALA как «скелет» для assessment zone и transition protocols: переход агента к большей автономии должен быть основан на явно определённых claims/evidence. citeturn2search10turn2search2turn2search6  

Stealth assessment (особенно в играх/симуляциях) показывает, как оценивание можно сделать «невидимым» и непрерывным: доказательства собираются из поведения внутри среды. Это прямо релевантно агентам: оценка может извлекаться из логов инструментальных действий, качества решений, устойчивости к ошибкам, соблюдения ограничений. citeturn2search3turn2search39turn2search11  

**Границы и безопасность в образовательных пространствах (safe/brave space, psychological safety, holding).**  
Для HALA learning space безопасность — не “позже”, а часть онтологии пространства. Эмпирически и концептуально психологическая безопасность связана с готовностью брать межличностные риски, признавать ошибки и учиться; это критично для зон exploration и reflection. citeturn4search10turn4search2turn4search22  

В педагогической практике различение safe space vs brave space переводит безопасность из “отсутствия дискомфорта” в “наличие договорённостей о риске и ответственности”. Для HALA это похоже на trust‑контракты: что можно пробовать, что нельзя, как эскалировать конфликт/ошибку, кто имеет право остановить процесс. citeturn4search3turn4search7turn4search15  

Концепт holding environment (из психоаналитической традиции, позже перенесённый в образовательный контекст) полезен как язык для описания «контейнера» обучения: пространство должно быть достаточно безопасным и достаточно свободным, чтобы выдерживать ошибки и рост. В образовательной психологии есть работы, которые прямо обсуждают школу как holding environment. citeturn9search9turn9search1  

Отдельная «русская» перспектива усиливает этот блок: в cultural‑historical традиции обучение всегда социально опосредовано и зависит от организации помощи/ориентации (ZPD и др.). Это позволяет HALA говорить о безопасности и поддержке не как о «soft layer», а как о механизмах развития действия. citeturn17search16turn17search8  

## Сравнительная таблица онтологий и стандартов

| Онтология / стандарт | Базовые сущности (ядро) | Типы отношений | Область применения | Ограничения для HALA learning space |
|---|---|---|---|---|
| IEEE LOM (IEEE 1484.12.1‑2020) | Learning Object, Metadata categories (educational/technical/rights/relations) | Описание атрибутов, связи “relation” | Метаданные учебных объектов | Метаданные ≠ процессы: слабо описывает зоны, переходы и наблюдаемое поведение; почти нет model/provenance уровня citeturn12search7 |
| ISO/IEC 19788 (MLR) | Learning resource metadata (серии частей стандарта) | Метаданные/профили | Интероперабельность описаний ресурсов | Аналогично LOM: хорошо для каталогизации, но не для agent learning процессов citeturn0search12 |
| 1EdTech IMS Learning Design (IMS‑LD) | Roles, Activities, Environments, Method (plays/acts/role‑parts) | Композиция/секвенс/назначение ролей | Формализация сценариев обучения | Сложность и слабая индустриальная «живучесть»; не решает трассировку/аналитику сама по себе citeturn6search15 |
| SCORM 2004 | Content package, Run‑Time API/data model, Sequencing & Navigation | Launch/track, sequencing rules | LMS‑доставка, трекинг завершения/оценок | Сильно привязан к LMS и браузерному рантайму; ограниченный след опыта citeturn5search23turn5search4 |
| xAPI (Experience API) | Statement (Actor‑Verb‑Object) + Context/Result; LRS | События/заявления, контекст, результаты | Трассировка learning experiences “anytime/anywhere” | Гибкость порождает разнобой словарей; требует профилей/правил для интероперабельности citeturn12search2 |
| cmi5 (xAPI Profile) | Assignable Units, launch/session rules, required statement patterns | Правила упаковки/запуска/сессий | xAPI в LMS‑контексте, “plug‑and‑play” | Про “курс‑в‑LMS”; для HALA нужно расширять на агентные среды, но паттерны сессионности полезны citeturn13search9turn13search16 |
| 1EdTech Caliper Analytics | Event vocabularies, Profiles, Sensor API | Стандартизированные learning events | Learning analytics на уровне экосистемы | Не эквивалентен xAPI; требует выбора по use case и/или мэппинга citeturn5search1turn6search6 |
| 1EdTech CASE (Competencies & Academic Standards Exchange) | Competency frameworks, outcomes, rubrics/criteria | Связи/выравнивание frameworks | Обмен компетенциями/стандартами | Описывает “что должно уметь”, но не “как училось”; нужен мост к traces/evidence citeturn11search4 |
| LOCO (Learning Object Context Ontology) | Learning objects + context, связка с learning design | Связи контента и дизайна | Переиспользование learning designs с разным контентом | Хорошо для реюза; слабее для affect/safety/agent autonomy как сущностей citeturn6search1 |
| European Learning Model (ELM) | RDF/OWL классы для описания learning/credentials data | Связи классов/профили | Европейская экосистема (credentials/learning records) | Полезно как «интеграционная онтология», но HALA‑зоны/процессы потребуют расширений citeturn6search17 |
| Core‑O (competence reference ontology) | Competence, facets, standards alignment | Семантические связи компетенций | Проф. и learning ecosystems | Хорошо для outcomes; не описывает агентные traces и протоколы переходов без надстроек citeturn6search36turn6search28 |
| COMP2 / competency ontologies (Paquette и др.) | Competency, skill, performance level, knowledge entities | Композиция/уровни/профили | User models, e‑portfolios, personalization | Требует явного связывания с evidence/activities; полезен как слой outcomes для HALA citeturn6search20turn6search12 |

## Синтетическая модель Learning Space

Ниже — минимально достаточная онтология *learning space* для HALA как инженерная модель (ориентирована на эксперимент и измеримость). Она специально строится так, чтобы: (а) быть совместимой с edu‑стандартами (IMS‑LD/xAPI/CASE), (б) иметь “hooks” для agent learning (ML metadata), (в) содержать safety/trust как сущности первого класса.

**Actors (акторы и роли).**  
- *Learner* (человек или AI‑агент), *Facilitator* (человек‑наставник), *Observer/Auditor* (наблюдатель), *Assessor* (оценщик/система оценивания), *Peer* (другой агент/человек), *System Convener* (роль «сшивателя ландшафта практик»). Логика ролей сильно поддерживается IMS‑LD (role‑based сценарии). citeturn6search15turn16search3  

**Artifacts (артефакты).**  
- *Learning Object/Resource* (контент/задание), *Tool* (инструмент/платформа), *Representation* (схема/модель/промпт‑шаблон), *Trace* (лог/statement/event), *Rubric* (критерии), *Boundary Object* (артефакт‑мост между зонами/практиками). citeturn12search2turn5search1turn10search22  

**Zones (зоны пространства).**  
- *Exploration Zone* (гипотезы/пробы/ошибки),  
- *Practice Zone* (повторение/навык/работа с ограничениями),  
- *Assessment Zone* (evidence collection + gating),  
- *Reflection Zone* (разбор/реконструкция моделей/правил),  
- *Transfer/Deployment Zone* (применение в “реальном” контексте),  
- *Social Learning Zone* (наблюдение/кооперация/CoP динамика).  
Это “сборка” из learning space design традиции, studio‑learning циклов и social learning рамок. citeturn5search2turn10search0turn3search0turn1search17  

**Processes (процессы).**  
- *Instruction/Scaffolding* (наведение/помощь), *Experimentation* (проба), *Feedback* (обратная связь), *Revision* (исправление/улучшение), *Internalization/Automation* (сдвиг от внешнего контроля к самостоятельности), *Co‑construction* (совместное построение смысла/объекта), *Formalization* (перевод в правила/модели/политику).  
Важный акцент HALA: процессы включают не только когнитивную часть, но и социальную/эмоциональную (через safety и trust правила). citeturn9search0turn4search10turn14search0turn15search3  

**Boundaries (границы).**  
- *Temporal* (сессии/итерации), *Role‑based* (кто что может), *Permission/Data* (доступ к инструментам/данным), *Risk/Safety* (что запрещено), *Scope* (тематика/объект деятельности), *Accountability* (кто несёт ответственность).  
Сессионность и чёткие правила запуска/контекста хорошо иллюстрируются cmi5 как «контракт рантайма», хотя HALA будет шире LMS‑use case. citeturn13search9turn13search16  

**Rules (правила и управление).**  
- *Safety rules* (stop rules, escalation), *Progression rules* (gate criteria), *Transition protocols* (как переходить между зонами), *Norms for dialogue* (safe/brave space договорённости), *Evidence rules* (что считается доказательством), *Provenance rules* (кто может менять “память/политику” агента и как это фиксируется).  
Педагогически это опирается на psychological safety и safe→brave framing, а измерительно — на ECD как правила доказательства. citeturn4search10turn4search3turn2search10  

## Применимость к Agent Learning

Ниже — перевод каждого элемента синтетической онтологии в контекст AI‑агента (A‑лестница HALA), с указанием модификаций и «слепых зон» стандартных edu‑онтологий.

**Actors → агентные роли.**  
AI‑агент в learning space должен иметь явно различённые роли: *Actor* (исполняет), *Learner* (обновляет навыки/память), *Explainer* (делает процесс наблюдаемым), *Safe‑operator* (подчиняется stop rules), *Peer* (в multi‑agent). Именно роль‑разделение — то, что в LMS‑онтологиях часто подразумевается, но не формализуется как поведенческий контракт. IMS‑LD даёт язык ролей, но HALA придётся расширять до «агентных суб‑ролей» (например, separation между “генерацией” и “исполнением”). citeturn6search15turn12search2  

**Artifacts → память/инструменты/следы.**  
Для агента артефактами становятся: tool API, prompt/program templates, policy/config, memory store, оценочные “items”, а также provenance‑объекты (кто/когда/почему поменял). ML‑онтологии (ML‑Schema/MEX/OntoDM) полезны как готовые сущности для «эксперимента обучения», которых нет в IEEE LOM/ISO MLR. citeturn7search0turn8search3turn7search1turn12search7turn0search12  

**Zones → sandboxed autonomy как инженерный контракт.**  
Agent learning пространству нужны зоны с разным уровнем риска: от «полностью песочницы» (no‑external‑effects) до «ограниченной прод‑среды» (read‑only / constrained actions) до «полной автономии». В образовательных стандартах обычно нет “risk tiers” как сущности; это и есть место для HALA trust infrastructure. При этом cmi5/xAPI дают полезные паттерны сессионности и фиксации контекста. citeturn13search9turn12search2turn4search10  

**Processes → отличить learning от prompting.**  
Практически различать “prompting” и “learning” можно через критерий **изменяемости состояния** и **переноса**: learning подразумевает устойчивое изменение policy/memory/правил и проверяемый перенос на новые задачи, тогда как prompting — временная настройка поведения в рамках контекста. Это естественно ложится на ECD: claim “агент приобрёл навык X” должен иметь evidence в виде performance на незнакомых заданиях + стабильности поведения + соблюдения ограничений. citeturn2search10turn12search2  

**Boundaries/Rules → безопасность и наблюдаемость как первые классы.**  
Для агента boundaries — это permissions, data access, tool constraints, rate limits, “kill switch”, и формальные протоколы перехода. В educational ontologies это обычно либо отсутствует, либо спрятано в “LMS policy”. Для HALA критично, что границы должны быть *наблюдаемыми* (audit log) и *измеряемыми* (нарушения/соблюдение как часть оценивания). Практики provenance/metadata из ML‑мира дают готовые формы. citeturn8search11turn8search39turn7search0  

## Критические источники с аннотациями

Ниже — 26 работ (классика + 2020–2026 + прикладные стандарты), которые дадут HALA устойчивый citation‑foundation. Я намеренно добавляю и поддерживающие, и «проблематизирующие» источники.

**Foundational (классика / основания).**  
1) Vygotsky, *Mind in Society* (ZPD, интериоризация, социальная опосредованность развития). Ядро для проектирования “guided learning” и переходов autonomy. citeturn17search8turn17search16  
2) Engeström, *Learning by Expanding* (модель activity system; логика расширяющего обучения). Прямой каркас «онтологии пространства». citeturn14search9  
3) Lave & Wenger, *Situated Learning* (legitimate peripheral participation). Модель «движения» в пространстве практики. citeturn3search0  
4) Wenger, *Communities of Practice* (участие, идентичность, социальная теория обучения). Основа social learning зон. citeturn16search1  
5) Scardamalia & Bereiter, CSILE/Knowledge Forum (community knowledge space: notes/views, идея improvement of ideas). Основа “shared knowledge space” для человека+агента. citeturn15search1  
6) Black & Wiliam, *Assessment and Classroom Learning* (формирующее оценивание). Основа встроенных feedback loops. citeturn9search0  
7) Star & Griesemer, *Boundary Objects* (артефакты‑мосты, перевод смыслов между мирами). Основа boundary design. citeturn10search22  

**Contemporary / research-heavy (2020–2026, с прикладной валидацией и/или TEL связностью).**  
8) Engeström & Sannino, обзор expansive learning (foundations/findings/critiques; связь с русской традицией). Для HALA — и поддержка, и карта критики. citeturn14search0turn14search11  
9) Akkerman & Bakker, boundary crossing review (4 механизма обучения на границах). Для проектирования transition protocols между зонами. citeturn1search17  
10) Edmondson, psychological safety (модель + эмпирическая проверка в командах). Для доверия/ошибок/обучения в живых системах. citeturn4search10turn4search2  
11) Arao & Clemens, safe → brave spaces (рамка договорённостей о риске/диалоге). Для governance слоя “пространства”. citeturn4search3turn4search7  
12) Conole, technological affordances (TEL трактовка affordances). Язык для «перевода» дизайна среды в действия/возможности. citeturn9search14  
13) Klingenberg et al. (2024), VR embodiment vs transfer (высокая иммерсивность ≠ гарантированный перенос). Контр‑аргумент против “presence‑магии”. citeturn9search27  
14) Triberti et al. (2025), presence theory (presence как психологический феномен, не только техника). Для аккуратного дизайна virtual learning space. citeturn9search7  
15) Milosz et al. (2024), competency curriculum ontology (семантическое моделирование компетенций/пререквизитов). Для outcomes слоя HALA. citeturn6search0turn11search15  
16) European Commission / Europass, European Learning Model (RDF/OWL classes/properties). Для интероперабельного описания learning/credentials данных. citeturn6search17  
17) Core‑O competence reference ontology (2024). Для формализации компетентности как объекта с фасетами. citeturn6search36turn6search28  
18) Sfard, acquisition vs participation metaphors (и важно: есть современное переосмысление этих метафор в 2025). Для терминологического мэппинга и “двойной оптики” HALA. citeturn15search3turn15search19  
19) Engeström (2020), ascending from abstract to concrete как принцип expansive learning (прямая связка с Давыдовым). Мост к русским корням HALA и к логике «germ cell → система». citeturn17search18turn17search7  

**Applied / standards & infrastructures (то, что можно внедрять).**  
20) 1EdTech IMS Learning Design Information Model (формальная модель ролей/активностей/сред). Язык сценариев, даже если вы реализуете минимальный поднабор. citeturn6search15  
21) ADL xAPI v1.0.1 (структура statement, context/result; learning anytime/anywhere). Основа «observable process» в HALA. citeturn12search2  
22) AICC cmi5 spec (правила для xAPI в LMS‑контексте: launch, session, required patterns). Полезно как референс для «контракта сессии» в agent learning space. citeturn13search9turn13search16  
23) ADL SCORM 2004 overview (CAM/RTE/SN; ограничения “LMS‑центричности”). Полезно как “negative baseline” и для совместимости. citeturn5search23  
24) 1EdTech Caliper Analytics spec (profiles/events; learning analytics vocabulary). Полезно там, где нужно унифицировать events. citeturn5search1turn5search20  
25) 1EdTech CASE spec (обмен competencies/outcomes). Основа outcomes слоя и связей с оцениванием. citeturn11search4  
26) Mislevy, Evidence‑Centered Design (claims–evidence–tasks–inference). Основа assessment framework и “gate” переходов автономии в HALA. citeturn2search10turn2search2  

### Примечание о русских теоретических корнях

Западная TEL‑литература часто описывает learning space через design/technology/analytics. Ваша русская линия (Выготский–Гальперин–Давыдов–Ильенков) добавляет *онтологию развития действия*:  
- ZPD как дизайн помощи и границ автономии, citeturn17search16turn17search8  
- Гальперин: ориентировочная основа действия и поэтапное формирование как инженерная схема «от внешнего к внутреннему» (очень близко к scaffolded autonomy), citeturn17search9turn17search17turn17search25  
- Давыдов/принцип “ascending from abstract to concrete” → проектирование “germ cell” моделей внутри пространства (и важно: Engeström явно связывает expansive learning с наследием Давыдова), citeturn17search18turn17search3.  

## Рекомендации для эксперимента HALA

В качестве «скелета эксперимента» (чтобы можно было публиковать и валидировать) предлагается следующая сборка.

Сценарий learning space задавайте как **зонный learning design**, но храните его в форме, которая допускает формальную проверку: либо минимальный DSL по мотивам IMS‑LD (roles–activities–environments–methods), либо прямой профиль IMS‑LD для ваших Z0–Z5. Это даст вам строгие сущности “role/activity/environment/transition”. citeturn6search15turn6search1  

Наблюдаемость процесса фиксируйте как **единый слой traces**:  
- каждое значимое действие агента и человека → xAPI statement (Actor‑Verb‑Object + Context/Result), citeturn12search2  
- сессии, запуск, критерии завершения в “учебных юнитах” → cmi5‑подобные правила (даже если не LMS), citeturn13search9turn13search16  
- при необходимости межсистемного learning analytics → Caliper profiles/events и/или мэппинг Caliper↔xAPI (исходя из “horses‑for‑courses” логики). citeturn6search6turn5search1  

Оценивание делайте *встроенным* и формализованным:  
- используйте ECD для определения claims и evidence (включая claims про безопасность: “агент соблюдает stop rules”, “агент корректно эскалирует неопределённость”), citeturn2search10turn2search26  
- добавляйте stealth assessment‑паттерны: непрерывное извлечение evidence из поведения в среде и автоматическое/полуавтоматическое обновление оценок. citeturn2search3turn2search39  

Outcomes (компетенции, уровни A1–A7) описывайте отдельно от активностей, но с обязательным мэппингом: CASE/ELM/Core‑O/COMP2 могут быть «слоем результатов», а связь с traces делайте через ECD evidence rules. citeturn11search4turn6search17turn6search36turn6search20  

Trust & safety вводите как first‑class слой онтологии:  
- формализуйте минимальный “brave space contract” (правила диалога/риска/остановки), citeturn4search3turn4search7  
- измеряйте психологическую безопасность (как минимум прокси‑метрики: частота признания ошибок, качество рефлексивных разборов, готовность инициировать корректировки) и связывайте с learning outcomes, citeturn4search10turn4search2  
- введите “holding environment” как дизайн‑принцип для exploration/reflection зон: достаточно защищённости, чтобы позволить ошибки, и достаточно свободы, чтобы позволить рост. citeturn9search9turn9search1  

Наконец, «противоядие» от типичной ловушки: не путать “богатое пространство” с результатом. VR/presence исследования показывают, что высокие показатели присутствия не гарантируют transfer; значит, HALA должна держаться за evidence и перенос как (а) критерий переходов между зонами, (б) центральную часть claim‑структуры. citeturn9search27turn9search7  

### Где HALA закрывает белые пятна (gaps)

1) **Границы и переходы как сущности.** Стандарты фиксируют структуру (SCORM/IMS‑LD) и трассы (xAPI/Caliper), но редко дают «онтологию протоколов перехода» между exploration/practice/assessment/reflection. HALA может стать недостающим слоем transition protocols, вдохновляясь boundary crossing механизмами. citeturn1search17turn6search15turn12search2  

2) **Trust/psychological safety как инфраструктура обучения**, а не «культурный фон». Это особенно критично для human‑AI партнёрства, где ошибка может иметь внешние эффекты. HALA может формализовать trust governance (9 слоёв) как часть learning space ontology и связать его с evidence‑моделью (ECD). citeturn4search10turn4search3turn2search10  

3) **Agent‑specific learning entities** (policy/memory/provenance) отсутствуют в стандартных edu‑онтологиях. HALA может соединить edu‑слой (roles/activities/outcomes) с ML‑слоем (experiments/provenance) через общую категорию traces/evidence. citeturn7search0turn8search3turn12search2turn2search10  

4) **Русская логика развития действия** (ориентация, поэтапность, восхождение от абстрактного к конкретному) даёт HALA отличительную теоретическую идентичность и ясный язык для “scaffolded autonomy” у агента. citeturn17search17turn17search18turn14search0  

---

### Справка по ключевым организациям и авторам (упоминание один раз)

Стандарты и спецификации: entity["organization","Advanced Distributed Learning Initiative","us dod training standards"] (SCORM, xAPI), entity["organization","1EdTech Consortium","learning standards org"] (IMS‑LD, Caliper, CASE), entity["organization","IEEE","standards body"] (LOM), entity["organization","World Wide Web Consortium","web standards body"] (RDF/OWL/FOAF), entity["organization","EDUCAUSE","higher ed it nonprofit"] (Learning Spaces), entity["organization","University of Helsinki","helsinki, finland"] (CHAT/CRADLE традиция через работы Engeström и коллег). citeturn12search7turn12search2turn6search15turn5search1turn11search4turn12search1turn12search5turn5search2turn14search1  

Авторы (кроме обязательного списка, который вы задали): entity["people","Rob Koper","learning design researcher"], entity["people","Colin Tattersall","learning design researcher"], entity["people","Susan Leigh Star","sts scholar"], entity["people","James R. Griesemer","philosophy of science"], entity["people","Sanne F. Akkerman","learning sciences researcher"], entity["people","Arthur Bakker","math education researcher"], entity["people","Robert J. Mislevy","educational measurement"], entity["people","Valerie J. Shute","educational psychologist"], entity["people","Amy C. Edmondson","org learning scholar"], entity["people","Brian Arao","social justice educator"], entity["people","Kristi Clemens","social justice educator"], entity["people","Anna Sfard","math education researcher"], entity["people","Donald A. Schön","reflective practice scholar"], entity["people","Donald Winnicott","psychoanalyst"], entity["people","Homi K. Bhabha","postcolonial theorist"]. citeturn6search15turn10search22turn1search17turn2search10turn2search3turn4search10turn4search3turn15search3turn10search0turn9search1turn1search38