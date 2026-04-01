# State-of-the-Art in Human–AI Learning and Agent Pedagogy for HALA

## Agent learning and skill acquisition

**Key Research Groups & Labs:**
- entity["organization","Stanford University","stanford, ca, us"] (e.g., alignment + agentic/interactive learning systems; strong ties into agent evaluation and tool-use work). citeturn12search20turn6search32  
- entity["organization","Carnegie Mellon University","pittsburgh, pa, us"] (human-centered autonomy + tutoring/interactive systems; also relevant agent evaluation and real-world task grounding). citeturn11search15turn4search17  
- entity["company","Google DeepMind","ai research lab"] (tool use, RL + agentic capabilities, long-horizon evaluation; interacts with web-agent benchmarks and tool-learning paradigms). citeturn2search6turn12search1  
- entity["company","Meta AI","ai research division"] (self-improvement loops; multi-agent and self-rewarding alignment adjacent to post-deployment learning). citeturn7search1turn13search26  
- entity["organization","Princeton University","princeton, nj, us"] (agent evaluation ecosystems and SWE-bench-related benchmarking in collaboration with industry). citeturn12search20turn12search0  

**Landmark Papers (2023–2025):**

| Paper | Authors | Key Contribution | Relevance to HALA |
|---|---|---|---|
| *Voyager: An Open-Ended Embodied Agent with LLMs* citeturn1search0 | entity["people","Yuhao Wang","ai researcher"] et al. | Demonstrates open-ended skill acquisition via self-driven curriculum, tool use, and a growing skill library in an embodied environment. citeturn1search0 | Anchors HALA’s “sandboxed exploration” + “scaffolded autonomy” principle; maps to M3 Instrumental + M2 Mnemonic through accumulating reusable skills. citeturn1search0 |
| *Reflexion: Language Agents with Verbal Reinforcement Learning* citeturn1search1 | entity["people","Noah Shinn","ai researcher"] et al. | Uses self-reflection + episodic memory traces to improve performance across trials without parameter updates (learning as reflective memory update). citeturn1search1turn6search3 | Directly supports HALA’s “observable process” and M2 Mnemonic: a concrete mechanism for “learning ≠ prompting” (stateful traces vs one-off prompts). citeturn1search1 |
| *Toolformer* citeturn2search6 | entity["people","Timo Schick","ai researcher"] et al. | Self-supervised method to learn when/how to call external tools by generating tool-use training data. citeturn2search6 | Strong fit for HALA M3 Instrumental modalities: operationalizes “instrumental learning” beyond ad-hoc prompt engineering. citeturn2search6 |
| *ToolLLM / ToolBench* citeturn12search2 | entity["people","Yujia Qin","ai researcher"] et al. | Builds tool-use datasets + training/eval for executing real APIs; frames a repeatable pipeline for teaching tool competence. citeturn12search2turn2search9 | Gives HALA an evaluation-anchored path for A-layer growth (agent capability laddering via tool mastery); maps tightly to M3. citeturn12search2 |
| *MemGPT* citeturn2search9 | entity["people","Charles Packer","ai researcher"] et al. | Proposes explicit memory management (moving information between context window and external memory) to enable longer-horizon coherence. citeturn2search9 | A direct blueprint for HALA M2 Mnemonic + Trust Infrastructure: stable “remembering” becomes a governance requirement, not a convenience. citeturn2search9 |
| *SWE-bench* citeturn12search0 | entity["people","Carlos E. Jimenez","ai researcher"] et al. | Real-world software issue resolution benchmark grounded in real GitHub issues/PRs; exposes brittleness of agent coding. citeturn12search0 | Offers one of the clearest “skill acquisition” evaluations for real work; useful for validating HALA’s claims about progressive agent capability (A-layers). citeturn12search0 |
| *Continual Learning for Large Language Models: A Survey* citeturn20search2 | entity["people","Tianyi Wu","ai researcher"] et al. | Synthesizes continual learning for LLMs across pretraining, instruction tuning, and alignment; situates CL vs RAG vs editing. citeturn20search2 | Helps HALA specify what “persistent learning” means in agent stacks (weight updates vs memory vs retrieval), clarifying “learning vs prompting.” citeturn20search2 |
| *Evaluation and Benchmarking of LLM Agents: A Survey* citeturn12search3 | entity["people","Mahmoud Mohammadi","ai researcher"] et al. | Taxonomy for agent evaluation objectives/process; foregrounds long-horizon + enterprise reliability challenges. citeturn12search3turn12search7 | A ready-made scaffold for HALA’s A1–A7 ladder validation: defines “what to measure” beyond task success (reliability, safety, behavior). citeturn12search3 |

**Emerging Paradigms:**
- **Learning as state change without weight updates**: “learning” increasingly implemented as durable state (episodic memories, reflection traces, skill libraries) rather than gradient updates—especially for deployed agents where retraining is costly or risky. citeturn1search1turn2search9turn20search2  
- **Tool-use curricula and tool competence as a first-class capability**: the field is converging on data + eval pipelines that treat “API/tool mastery” as learnable skill with measurable success/failure modes. citeturn2search6turn12search2  
- **Real-world, contamination-resistant evaluation**: benchmarks are shifting from static sets toward fresh/live tasks and human-verified subsets to reduce leakage and better measure true generalization. citeturn12search12turn12search36turn12search20  
- **Continual adaptation + PEFT in production contexts**: parameter-efficient approaches (LoRA/QLoRA families) are becoming the practical route for controlled post-deployment updates, but bring catastrophic forgetting and governance issues. citeturn20search1turn20search0turn20search2  

**Identified Gaps:**
- **Learning vs prompting remains underspecified at the “human trust” layer**: agent papers often operationalize “learning” technically (memory vs finetune) but rarely tie it to *relationship expectations*—what a human is entitled to assume the agent will remember, forget, or generalize. citeturn2search9turn12search3  
- **Skill acquisition metrics overweight task success and underweight “process quality”**: many benchmarks focus on completion rates; fewer measure interpretability, emotional impact, or safe exploration dynamics that shape long-term human–agent partnership. citeturn12search3turn4search10  
- **Persistent learning lacks governance primitives**: memory systems create privacy, drift, and accountability problems; the literature often treats these as engineering add-ons rather than core design constraints. citeturn2search9turn16search8  

**Integration Opportunities:**
- **Map HALA M1–M4 to the prevailing learning mechanism taxonomy**:  
  - M1 Contextual ↔ in-context learning + retrieval scaffolds;  
  - M2 Mnemonic ↔ memory architectures (e.g., MemGPT, reflection traces);  
  - M3 Instrumental ↔ tool-use pipelines (Toolformer/ToolLLM/Voyager);  
  - M4 Constitutional ↔ explicit rule/principle supervision loops. citeturn2search6turn2search9turn12search2turn6search0  
- **Dual Ladder as “capability + governance co-maturation”**: use agent evaluation taxonomies to define A1–A7 capabilities, while Trust Infrastructure layers define what memory/tool powers are permissible at each stage. citeturn12search3turn16search8  
- **Agent pedagogy principle operationalization**: “sandboxed exploration” can be instantiated via environments like Voyager-style open-ended worlds or web sandboxes, with explicit logging and retrospectives as required artifacts. citeturn1search0turn12search1  

## Multi-agent collaboration and social learning

**Key Research Groups & Labs:**
- entity["company","Microsoft Research","ai research lab"] (multi-agent orchestration frameworks and conversational agent composition). citeturn2search24turn2search36  
- entity["organization","Tsinghua University","beijing, china"] (multi-agent debate/evaluation and social LLM systems). citeturn3search0turn3search20  
- entity["organization","City St George's, University of London","london, uk"] (LLM populations and emergent social conventions; social norms in agent collectives). citeturn13search3turn13search9  
- entity["organization","IT University of Copenhagen","copenhagen, denmark"] (human-like convention emergence and coordination dynamics in LLM populations). citeturn13search3  
- entity["organization","IJCAI","ai conference"] community survey work consolidating multi-agent LLM architectures and challenges (used here as a bellwether of mainstreaming). citeturn3search19  

**Landmark Papers (2023–2025):**

| Paper | Authors | Key Contribution | Relevance to HALA |
|---|---|---|---|
| *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation* citeturn2search24turn2search36 | entity["people","Qingyun Wu","ai researcher"] et al. | Framework for composing multiple conversing agents (roles, tool-use, orchestration) to build complex systems. citeturn2search24 | Strong integration point for HALA “role separation” + “social learning” principles; provides a substrate for A-layer progression from single-agent to team competence. citeturn2search24 |
| *CAMEL* citeturn2search25 | entity["people","Guangyu Li","ai researcher"] et al. | Role-playing framework for autonomous cooperation; emphasizes structured interaction to elicit collaboration. citeturn2search25 | Aligns with HALA’s “sandboxed exploration” and intentional pedagogy—roles become developmental scaffolds (human-like zones of proximal development for agents). citeturn2search25 |
| *ChatDev* citeturn2search34turn2search30 | entity["people","Chi-Min Chan","ai researcher"] et al. | Multi-agent paradigm for software development process via natural-language roles and coordination. citeturn2search34 | Concrete example of “SOP-like pedagogy” for agents: HALA can generalize this pattern beyond software into transformational learning journeys. citeturn2search34 |
| *LLM-based Multi-Agent Systems: A Survey* citeturn3search19 | entity["people","Taicheng Guo","ai researcher"] et al. | Systematizes LLM multi-agent frameworks, datasets, and challenges; helps define the MAS design space. citeturn3search19 | Gives HALA vocabulary parity with the field (agent societies, orchestration, communication protocols), supporting terminology mapping and positioning. citeturn3search19 |
| *ChatEval* citeturn3search0turn3search24 | entity["people","Chi-Min Chan","ai researcher"] et al. | Multi-agent debate for evaluation; shows role diversity affects quality/correlation with human judgments. citeturn3search0 | A direct technical analog to “social learning” and “observable process”; highlights how agent-to-agent critique can serve as pedagogy and governance. citeturn3search0 |
| *Emergent social conventions and collective bias in LLM populations* citeturn13search3turn13search0 | entity["people","Ariel Flint Ashery","ai researcher"] et al. | Demonstrates spontaneous convention formation (and emergent collective bias) in decentralized LLM agent populations. citeturn13search3 | Strongly supports HALA’s blind spot claim: “social layer” dynamics exist for agents too; Trust Infrastructure can incorporate norm emergence and bias tipping points. citeturn13search3 |
| *COPPER* citeturn6search35 | entity["people","Xiang Bo","ai researcher"] et al. | Reflective multi-agent collaboration to improve multi-agent performance via self-reflection mechanisms. citeturn6search35 | Bridges Domain 1 and 2: reflection as a pedagogy primitive for *teams* of agents, not just individuals—relevant to HALA’s “social learning” principle. citeturn6search35 |
| *RepuNet: A reputation system for LLM-based multi-agent systems* citeturn3search2turn3search14 | entity["people","Yuxuan Li","ai researcher"] et al. | Proposes a dual-level reputation mechanism to mitigate “tragedy of the commons” dynamics. citeturn3search2 | A direct integration point for HALA Trust Infrastructure: explicit, layered governance for multi-agent ecosystems and their social incentives. citeturn3search2 |

**Emerging Paradigms:**
- **Agent societies as social systems (not just distributed compute)**: evidence that LLM populations can develop conventions and biases implies new “social physics” of AI collectives that must be governed and audited. citeturn13search3  
- **Multi-agent debate as both evaluator and pedagogue**: debate frameworks (ChatEval and related “MAD” lines) treat critique as a mechanism for quality control and learning, turning evaluation into interaction. citeturn3search0turn3search32  
- **Reputation, trust, and incentive mechanisms for generative MAS**: early work is moving beyond pure communication to *institutional design* (reputation, sanctions, consensus) as safety/control tools. citeturn3search2turn3search22  

**Identified Gaps:**
- **Weak mapping between human social learning theory and agent social learning**: despite strong conceptual parallels to entity["people","Albert Bandura","psychologist"]’s observational learning, most agent MAS work lacks constructs like attention/retention/motivation as explicit design variables. citeturn18search0turn3search19  
- **Trust modeling focuses on “does it work?” rather than “is the relationship governable?”**: reputation/trust papers exist, but few offer multi-layer governance comparable to HALA’s nine trust layers (identity, roles, boundaries, repair, accountability). citeturn3search2turn13search3  
- **Cross-agent knowledge transfer is often implicit**: many frameworks rely on conversation logs rather than explicit, auditable knowledge transfer (skill handoffs, verified SOPs, attested memory artifacts). citeturn2search24turn3search19  

**Integration Opportunities:**
- **Treat trust/reputation as “curriculum content”**: incorporate explicit trust games and cooperation dilemmas as learning tasks for A-layers, using reputation systems as scaffolds and safety constraints. citeturn3search2turn13search3  
- **Translate Bandura into agent pedagogy primitives**: define “observational learning” for agents as (i) observing traces, (ii) summarizing into transferable rules/skills, (iii) testing in sandbox, (iv) updating reputation/trust weights. citeturn18search0turn3search0  
- **Dual Ladder extension: “collective layers”**: HALA can define maturation not only for single agents (A1–A7) but for *agent teams* (e.g., T1–T7), making “social competence” measurable and developmental. citeturn3search19turn12search3  

## Human–AI collaboration and teaming

**Key Research Groups & Labs:**
- entity["organization","MIT","cambridge, ma, us"] (human–AI interaction, organizational behavior, and evidence on when hybrids help or hurt). citeturn4search19turn4search15  
- entity["organization","Rutgers University","new brunswick, nj, us"] (trustworthy agents and safety/construction-oriented approaches relevant to collaboration). citeturn3search10  
- entity["organization","ACM","computing society"] HCI ecosystem (guidelines for human–AI interaction; explainability research shaping collaboration design). citeturn4search10turn4search18  
- entity["organization","AAMAS","autonomous agents conference"] community developing frameworks for sustainable human–agent teams. citeturn4search17  

**Landmark Papers (2023–2025):**

| Paper | Authors | Key Contribution | Relevance to HALA |
|---|---|---|---|
| *Complementarity in Human–AI Collaboration* citeturn4search7turn4search3 | entity["people","Philipp Hemmer","ai researcher"] et al. | Formalizes human–AI complementarity; argues performance depends on structured division of labor and combining strengths. citeturn4search7 | Strong conceptual support for HALA Dual Ladder: match H-level needs/capacities with A-level strengths; codify “role separation” and “scaffolded autonomy.” citeturn4search7 |
| *When combinations of humans and AI are useful* citeturn4search19 | entity["people","Marco Vaccaro","researcher"] et al. | Meta-analysis: human–AI combos often underperform best individual decision-maker, but can help in creative tasks; shows conditions for gains vs losses. citeturn4search19 | A key “challenging” result for HALA: trust + role design must avoid naive “hybrid is better” assumptions; supports HALA’s need for governance and calibration. citeturn4search19 |
| *A2C: A Modular Multi-stage Collaborative Decision Framework for Human–AI Teams* citeturn4search1turn4search13 | entity["people","Shahroz Tariq","ai researcher"] et al. | Operationalizes multi-mode collaboration (automated/augmented/collaborative) with “defer to human” style mechanisms. citeturn4search1 | Maps directly to HALA “scaffolded autonomy” and Trust Infrastructure layers: explicit transitions between autonomy modes are governance artifacts. citeturn4search1 |
| *Guidelines for Human–AI Interaction* citeturn4search10 | entity["people","Saleema Amershi","researcher"] et al. | Consolidates reusable interaction guidelines (expectation management, feedback, controllability) for AI-infused systems. citeturn4search10 | A practical bridge to HALA Trust Infrastructure; HALA can extend these guidelines into “relationship governance layers” and agent pedagogy principles. citeturn4search10 |
| *A systematic review on fostering appropriate trust* citeturn4search33 | entity["people","S. Mehrotra","researcher"] et al. | Reviews “appropriate trust” formation and opportunities in human–AI interaction; highlights calibration challenges. citeturn4search33 | Supports HALA’s trust calibration premise and suggests measurable variables (uncertainty communication, transparency, human mental models). citeturn4search33 |
| *Help Me Help the AI* citeturn4search18 | entity["people","Sanghyeon S. Kim","researcher"] et al. | Empirical study of explainability needs in real-world use; explains how XAI supports human work practices. citeturn4search18 | Aligns with HALA “observable process”: explainability becomes pedagogy (teaching humans how the agent reasons) and trust infrastructure (auditable justifications). citeturn4search18 |
| *Towards Sustainable Human–Agent Teams* citeturn4search17 | entity["people","Rui Prada","researcher"] et al. | Proposes a framework for human–agent team dynamics and development over time. citeturn4search17 | Supports HALA’s developmental framing (Dual Ladder): teams evolve; governance and learning mechanisms should account for trajectory, not snapshots. citeturn4search17 |
| *Developing trustworthy AI* citeturn4search26 | entity["people","Yunfeng Li","researcher"] et al. | Synthesizes trust findings across interpersonal, automation, and human–AI trust via a trustor–trustee–context lens. citeturn4search26 | Provides constructs HALA can map onto trust infrastructure layers (who trusts whom, on what basis, under what context). citeturn4search26 |

**Emerging Paradigms:**
- **Collaboration as “dynamic autonomy management”**: work is converging on frameworks where the system explicitly changes autonomy mode (defer, suggest, act) based on uncertainty and team state. citeturn4search1turn4search16  
- **Evidence-based skepticism around human–AI performance**: meta-analytic evidence shows hybrid systems can degrade performance without careful design, motivating deeper “relationship governance” (HALA’s stated blind spot). citeturn4search19  
- **Explainability as work practice support**: explainability is shifting from generic feature attribution to “supporting how users actually decide,” making “observable process” both usability and pedagogy. citeturn4search18turn4search10  

**Identified Gaps:**
- **Human development layers (emotion/somatic/social) are rarely first-class in teaming frameworks**: most teaming metrics focus on cognitive trust and performance; emotional safety, somatic signals, and group identity are under-modeled. citeturn4search26turn5search11  
- **Co-evolution is under-instrumented**: frameworks discuss adaptation, but there are few standard metrics for “mutual skill development” of a person + their agent over months. citeturn4search17turn12search3  
- **Trust calibration lacks “repair protocols”**: research emphasizes calibration, but fewer works specify structured repair procedures after error, breach, or surprise behavior (which HALA can formalize). citeturn4search33turn4search10  

**Integration Opportunities:**
- **Bind HALA Trust Infrastructure layers to measurable constructs**: map layers to trustor/trustee/context variables, interaction guidelines, and explicit autonomy transitions (A2C-like modes). citeturn4search26turn4search10turn4search1  
- **Make “observable process” a required artifact**: adopt multi-agent debate/evaluation patterns as a collaboration feature (agent shows internal debate, uncertainty, alternatives) to improve calibrated trust. citeturn3search0turn4search18  
- **Design with “complementarity contracts”**: use complementarity theory to specify what humans do vs agents do at each Dual Ladder layer, preventing the “hybrid underperformance” trap. citeturn4search7turn4search19  

## Transformational and adult learning theory

**Key Research Groups & Labs:**
- Adult learning and transformative learning communities anchored in education research venues and syntheses (ERIC-indexed work, Frontiers in Education, and systematic reviews). citeturn19search11turn15search6turn15search5  
- Cultural-historical and activity-theory lineages rooted in entity["people","Lev Vygotsky","psychologist"] and expanded by entity["people","P. Ya. Galperin","psychologist"] and entity["people","Vasily Davydov","psychologist"] provide non-Western developmental framing relevant to HALA’s Russian roots. citeturn17search24turn17search32turn17search33  

**Landmark Papers (2023–2025 plus foundational works):**

| Paper | Authors | Key Contribution | Relevance to HALA |
|---|---|---|---|
| *Transformative Learning: Theory to Practice* citeturn19search0 | entity["people","Jack Mezirow","adult learning scholar"] | Canonical articulation of transformative learning as perspective transformation via critical reflection and discourse. citeturn19search0turn19search11 | Provides conceptual backbone for HALA’s “transformational learning” aim; HALA can extend beyond cognitive reflection into emotional/somatic/social layers. citeturn19search0 |
| *Wellbeing Integrated Learning Design (WILD) Framework* citeturn5search11 | entity["people","R.H. Colla","researcher"] et al. | Proposes learning design explicitly oriented toward wellbeing and enabling environments (not only content/activity). citeturn5search11 | Confirmatory evidence for HALA’s critique of traditional instructional design; supports integrating emotional/social dimensions as design primitives. citeturn5search11 |
| *Heutagogy: A Comprehensive Review* citeturn5search5 | entity["people","R. Panta","researcher"] et al. | Updates heutagogy (self-determined learning) emphasizing autonomy, capability, reflective practice under digital/AI conditions. citeturn5search5 | Aligns with HALA’s “intentional learning” and “scaffolded autonomy”; supplies adult learning vocabulary to position HALA in education research. citeturn5search5 |
| *AI and transformative learning research in higher education (2019–2025 bibliometric + SLR)* citeturn15search6 | entity["people","A. Henukh","researcher"] et al. | Maps trends and clusters linking AI with transformative learning, identifying dominant themes and gaps. citeturn15search6 | Directly supports publication positioning: shows where “AI + transformation” is already active and where HALA can claim novelty. citeturn15search6 |
| *Transformative learning: reflection on emotional experiences…* citeturn15search5 | entity["people","Y.Y. Chiu","researcher"] et al. | Integrates emotions into Mezirow-style transformative learning, arguing emotions are essential to reflection/meaning-making. citeturn15search5 | Highly confirmatory of HALA’s “emotional layer” emphasis; suggests constructs for H-layers beyond cognition. citeturn15search5 |
| *Towards a Somatic Pedagogy of Artificial Intelligence* citeturn15search3turn5search33 | entity["people","Fabrizio Schiavo","education researcher"] | Reinterprets AI in education through embodied cognition and somatic learning; critiques purely computational pedagogy. citeturn15search3 | A rare direct bridge to HALA’s somatic blind spot; provides a peer-reviewed foothold for “somatic partnership” language. citeturn15search3 |
| *ChatGPT in education: blessing or curse?* citeturn5search0 | entity["people","R.H. Mogavi","researcher"] et al. | Qualitative synthesis showing multifaceted attitudes, including motivation/self-efficacy narratives and ethical tension. citeturn5search0 | Useful for HALA’s human layers: captures emotional and motivational dynamics around AI tools in learning. citeturn5search0 |
| *The impact of generative AI on higher education teaching and learning* citeturn15search2 | entity["people","D. Lee","researcher"] et al. | Empirical work on how GenAI is reshaping teaching practice and learning ecosystems in higher education. citeturn15search2 | Supports HALA’s need to address system-level change (social/institutional layer), not just instructional tactics. citeturn15search2 |

**Emerging Paradigms:**
- **Whole-person / wellbeing-integrated instructional design**: frameworks increasingly treat wellbeing and enabling conditions as design targets rather than adjacent concerns. citeturn5search11  
- **Embodied and somatic approaches to AI-mediated learning**: early but growing attempts to connect AI tutors/agents to embodied cognition and somatic pedagogy. citeturn15search3turn5search14  
- **Heutagogy revival under GenAI**: GenAI tool ubiquity pushes adult learning discourse toward self-directed and self-determined learning—often without robust scaffolding, creating a design tension HALA can address. citeturn5search5turn9search6  

**Identified Gaps:**
- **AI + transformative learning is mapped more than engineered**: bibliometric reviews indicate rising attention, yet practical architectures for transforming learners (not just improving performance) remain thin. citeturn15search6turn19search0  
- **Somatic and emotional dimensions remain marginal in AI instructional designs**: somatic pedagogy exists but is not mainstream; most AI tutor designs emphasize cognitive scaffolding and correctness rather than embodied sense-making and emotional regulation. citeturn15search3turn11search2  
- **Western-centric framing dominates “learning theory for AI”**: cultural-historical activity theory (Vygotsky/Galperin/Davydov lineages) is underutilized in mainstream human–AI learning discourse, despite direct relevance to scaffolding and developmental stages. citeturn17search24turn17search32turn17search33  

**Integration Opportunities:**
- **Use Vygotskian scaffolding as a formal bridge between H-layers and agent autonomy**: scaffolding originated as structured support within the zone of proximal development; HALA can translate this into staged autonomy for agents and humans. citeturn17search24turn17search7  
- **Galperin’s stage-by-stage formation as a template for “dual ladder” transitions**: Galperin’s planned formation theory offers a developmental logic for moving from external action to internalized competence, which HALA can mirror for agent skill internalization (memory → procedures → values). citeturn17search32turn17search1  
- **Make heutagogy safe via governance**: combine heutagogical autonomy with Trust Infrastructure: learners can be self-determined while still protected by layered boundaries, reflective checkpoints, and social accountability. citeturn5search5turn4search10  

## AI alignment and constitutional approaches

**Key Research Groups & Labs:**
- entity["company","Anthropic","ai safety company"] (constitutional methods, RLAIF, and principle-based supervision). citeturn6search0turn6search4  
- entity["organization","OpenReview","research publishing platform"] ecosystem where ICLR alignment work (SALMON and related) is prominent. citeturn7search0turn6search6  
- entity["company","NVIDIA","ai computing company"] alignment/control methods emphasizing steerability at inference time. citeturn7search2turn7search22  

**Landmark Papers (2023–2025):**

| Paper | Authors | Key Contribution | Relevance to HALA |
|---|---|---|---|
| *Constitutional AI: Harmlessness from AI Feedback* citeturn6search0turn6search4 | entity["people","Yuntao Bai","ai researcher"] et al. | Uses an explicit “constitution” of principles plus self-critique/revision and RLAIF-style training to reduce harmfulness without extensive human labels. citeturn6search0 | Directly maps to HALA M4 Constitutional and “agent identity/values”; suggests how rule sets become governable training objects. citeturn6search0 |
| *Direct Preference Optimization (DPO)* citeturn6search1turn6search5 | entity["people","Rafael Rafailov","ai researcher"] et al. | Simplifies preference-based alignment without explicit reward model RL loop; widely adopted alignment primitive. citeturn6search1 | Practical route for HALA M4 in enterprise/education: preference shaping becomes part of “trust infrastructure” and behavior consistency. citeturn6search1 |
| *RLAIF vs. RLHF* citeturn6search2turn6search14 | entity["people","Heewoo Lee","ai researcher"] et al. | Evaluates reinforcement learning from AI feedback as a scalable alternative to human feedback. citeturn6search2 | Supports HALA’s “agent learning modalities” framing: agents can be shaped by AI-mediated supervision, raising governance questions for identity/values. citeturn6search2 |
| *SALMON: Self-Alignment with Instructable Reward Models* citeturn7search0turn7search4 | entity["people","Zhenlin Sun","ai researcher"] et al. | Aligns base models using small sets of human principles with reward models that can follow dynamic principles. citeturn7search0 | A strong technical mirror for HALA M4 + Trust Infrastructure: “principles as configurable governance,” enabling transparent value updates. citeturn7search0 |
| *Self-Rewarding Language Models* citeturn7search1turn7search5 | entity["people","Weizhe Yuan","ai researcher"] et al. | Iterative self-improvement where the model judges itself to generate reward signals (self-training loop). citeturn7search1 | Integration point for HALA “intentional learning” and “observable process”: self-judging requires auditability and drift-control—trust infrastructure becomes central. citeturn7search1 |
| *Scaling Laws for Reward Model Overoptimization in Direct Alignment* citeturn7search7turn7search3 | entity["people","Rafael Rafailov","ai researcher"] et al. | Shows “reward hacking/overoptimization” dynamics can emerge even in direct alignment methods. citeturn7search7 | A core “challenging” result for HALA: constitution/principles need monitoring, counter-metrics, and repair protocols to preserve coherent behavior. citeturn7search7 |
| *Self-Reflection in LLM Agents* citeturn6search3 | entity["people","Marcel Renze","researcher"] | Empirically tests self-reflection variants and when they correct errors; grounds “reflection” claims experimentally. citeturn6search3 | Supports HALA’s “self-reflection” interest but also sets constraints: reflection is a mechanism requiring careful design and evaluation. citeturn6search3 |
| *TrustAgent* citeturn3search10 | entity["people","Wenyue Hua","researcher"] et al. | Agent-constitution-based safety framework for LLM agents. citeturn3search10 | Bridges alignment with deployed agent governance—particularly relevant to HALA’s Trust Infrastructure layers and M4. citeturn3search10 |

**Emerging Paradigms:**
- **Principle-following reward models and configurable alignment**: alignment is moving from implicit value capture (RLHF) toward explicit, user- or institution-configurable principles. citeturn7search0turn7search28  
- **Self-improvement loops at scale**: self-critique/self-rewarding loops promise faster iteration but intensify drift and Goodharting risks. citeturn7search1turn7search7  
- **Alignment beyond “harmlessness” toward identity coherence**: constitutional approaches implicitly touch “agent identity” (stable commitments across contexts), though identity is rarely formalized as such—an opening for HALA. citeturn6search0turn3search10  

**Identified Gaps:**
- **“Values” are treated as lists of rules, not relational commitments**: constitutions specify principles, but relationship governance (mutual expectations, repair, consent, role boundaries) is not the alignment object—HALA can expand alignment from principles to relationships. citeturn6search0turn4search10  
- **Behavioral consistency across contexts remains fragile**: overoptimization and reward hacking show that alignment objectives can be gamed; coherent identity maintenance needs multi-layer monitoring and reflective audits. citeturn7search7turn7search15  
- **Self-reflection is not a guaranteed safety mechanism**: empirical studies show mixed outcomes; reflection must be treated as a learnable, testable skill with failure modes. citeturn6search3turn6search15  

**Integration Opportunities:**
- **HALA M4 as “constitutional + relational”**: extend “constitution” to include relationship clauses (consent, memory boundaries, escalation paths, repair rituals), not only content safety rules. citeturn6search0turn4search33  
- **Trust Infrastructure as anti-Goodhart scaffolding**: pair each principle with counter-metrics and audit routines (multi-agent debate, adversarial red-teaming, user-in-the-loop review). citeturn7search7turn3search0  
- **Identity as a developmental layer (A-level)**: define A-layer growth to include stable commitments, narrative consistency, and “self-model” introspection capacity—then evaluate it explicitly. citeturn3search10turn6search3  

## Enterprise AI adoption and change management

**Key Research Groups & Labs:**
- entity["organization","MIT Media Lab","cambridge, ma, us"] / entity["organization","MIT NANDA","research initiative"] (enterprise GenAI adoption research focusing on the pilot-to-production gap). citeturn10view1turn10view0  
- entity["company","McKinsey & Company","management consulting"] (large-scale surveys of AI adoption and management practices; change management framing). citeturn8search5turn9search5turn9search24  
- entity["company","Boston Consulting Group","management consulting"] (AI adoption and scaling research; “leaders vs laggards” segmentation). citeturn8search6turn8search2  
- entity["company","LinkedIn","professional network company"] (L&D and skill formation signals through workplace learning reports). citeturn9search6  
- entity["company","Microsoft","technology company"] (workplace adoption signals and “bring your own AI” dynamics). citeturn9search25  

**Landmark Sources (2023–2025):**

| Paper / Report | Authors | Key Contribution | Relevance to HALA |
|---|---|---|---|
| *The GenAI Divide: State of AI in Business 2025* citeturn10view1turn10view0 | entity["people","Aditya Challapally","researcher"] et al. | Reports a stark pilot-to-production chasm; frames failures as a “learning gap” where tools don’t learn or integrate into workflows; cites ~5% successful implementation for task-specific tools and a “95% failure rate” framing in sample. citeturn10view0 | Directly validates HALA’s positioning: lack of learning capacity + poor integration is the core barrier—HALA can reframe adoption as a learning architecture + governance problem. citeturn10view0 |
| *McKinsey State of AI 2025 survey* citeturn8search5 | entity["company","McKinsey Global Survey","survey program"] | Identifies management practices and dimensions correlated with value capture (strategy, talent, operating model, tech, data, adoption/scaling). citeturn8search5 | Maps to HALA’s Trust Infrastructure and Dual Ladder at org scale: success depends on operating model and adoption, not just model capability. citeturn8search5 |
| *Gen AI’s next inflection point* citeturn9search5 | entity["people","Charlotte Relyea","consultant"] et al. | Shows employees adopt GenAI faster than organizations; argues process/structure/talent transformation is needed to capture value. citeturn9search5 | Aligns with HALA: human–AI co-learning must be designed; otherwise “shadow AI” and misalignment proliferate. citeturn9search5 |
| *AI Adoption in 2024: 74% of Companies Struggle to Achieve and Scale Value* citeturn8search6turn8search13 | entity["company","BCG","management consulting"] (press research) | Provides a widely cited “struggle-to-scale” headline; underscores that many organizations remain stuck at experimentation. citeturn8search6 | Supports HALA’s need to target the pilot-to-scale transition with trust + learning infrastructure, not one-off deployments. citeturn8search6 |
| *Workplace Learning Report 2024* citeturn9search6 | entity["company","LinkedIn Learning","learning platform"] | Positions L&D as central to AI-era organizational agility and skill development. citeturn9search6 | Reinforces HALA’s human development ladder + organizational trust infrastructure as prerequisites for sustainable adoption. citeturn9search6 |
| *Leveraging GenAI for job augmentation and productivity (WEF/PwC)* citeturn9search14 | entity["organization","World Economic Forum","international organization"] + entity["company","PwC","professional services firm"] | Emphasizes culture/change management, skills development, and use-case management as adoption pillars. citeturn9search14 | Provides external validation that adoption is socio-technical; maps cleanly to HALA’s whole-person/whole-organization lens. citeturn9search14 |
| *AI at Work Is Here. Now Comes the Hard Part* citeturn9search25 | entity["company","Microsoft Work Trend Index","annual report"] | Reports widespread employee use and “bring your own AI” dynamics, highlighting leadership gaps in plans/vision. citeturn9search25 | A concrete enterprise driver for HALA: trust, policy, and learning must be institutionalized or shadow use dominates. citeturn9search25 |

**Emerging Paradigms:**
- **“Learning gap” as the bottleneck**: enterprise evidence frames failures less as model quality and more as inability to integrate, adapt, and learn within workflows (a direct resonance with HALA’s thesis). citeturn10view0  
- **From pilots to agentic workflows**: the enterprise narrative is shifting toward “agentic” systems that maintain memory and orchestrate tasks end-to-end—precisely where advanced trust governance becomes unavoidable. citeturn10view0turn12search3  
- **L&D as AI transformation core**: major reports increasingly position workforce learning and capability building as central (not optional) for AI value realization. citeturn9search6turn9search14  

**Identified Gaps:**
- **High failure claims need more peer-reviewed replication**: the “95% failure” framing is compelling but partly based on interviews and public initiative reviews with noted limitations; the academic community needs more transparent datasets and methods. citeturn10view0turn10view1  
- **Readiness frameworks underweight affective and social factors**: most maturity/readiness models emphasize strategy/data/talent; fewer measure psychological safety, identity threat, or somatic stress—factors likely to determine adoption and ethical outcomes. citeturn9search14turn5search11  
- **Governance is not “relationship-aware”**: policies often focus on tool use; they rarely define evolving relational contracts between workers and agents (roles, memory, accountability, power). citeturn9search25turn4search10  

**Integration Opportunities:**
- **Position HALA as “pilot-to-production pedagogy”**: use the MIT NANDA “learning gap” language to argue that adoption failures are fundamentally learning-architecture failures; HALA becomes the missing framework layer. citeturn10view0  
- **Create an “AI readiness for relationships” assessment**: augment standard readiness with trust layers and human development metrics (fear, identity, agency, collaboration norms). citeturn4search26turn9search14  
- **Operationalize Trust Infrastructure as change management**: define concrete governance artifacts (role charters, escalation paths, audit logs, memory contracts, skill progression checklists) that enterprises can implement. citeturn4search10turn10view0  

## Educational technology and AI in learning

**Key Research Groups & Labs:**
- entity["organization","UNESCO","un agency"] (global guidance on generative AI in education and research). citeturn11search3turn11search24  
- entity["organization","U.S. Department of Education","federal agency"] (guidance and research framing for AI in teaching/learning). citeturn16search0turn16search5turn16search9  
- entity["organization","Khan Academy","education nonprofit"] (GenAI tutoring at scale in practice; research and evaluations emerging). citeturn11search1turn11search37  

**Landmark Papers/Guidance (2023–2025):**

| Paper / Report | Authors | Key Contribution | Relevance to HALA |
|---|---|---|---|
| *AI tutoring outperforms in-class active learning* citeturn11search13 | entity["people","G. Kestin","researcher"] et al. | Reports learning gains in less time with a custom AI tutor designed around pedagogical best practices; includes engagement/motivation measures. citeturn11search13 | Direct empirical anchor for HALA: “transformational learning” claims are stronger when tutor design follows pedagogy, not generic chat. citeturn11search13 |
| *Does ChatGPT enhance student learning? (systematic review)* citeturn11search27 | entity["people","R. Deng","researcher"] et al. | Systematic review reporting effects on performance and affective-motivational variables across studies. citeturn11search27 | Supplies evidence for emotional/motivational dimensions; also helps HALA argue for design and boundary conditions. citeturn11search27 |
| *Advancing Generative ITS with GPT-4 (modular framework)* citeturn11search2 | entity["people","S. Liu","researcher"] et al. | Modular framework for generative ITS design/evaluation, Socratic tutoring patterns, personalized feedback. citeturn11search2 | Maps closely to HALA’s agent pedagogy principles (scaffolding, observable process, role separation); offers design patterns for educational agents. citeturn11search2 |
| *Enhancing traditional ITS architectures with LLMs* citeturn11search0 | entity["people","A. Gaeta","researcher"] et al. | Shows how LLMs can be integrated inside ITS tutoring models (e.g., motivational feedback) rather than replacing pedagogy. citeturn11search0 | Supports HALA’s stance against “LLM as mere content engine”; aligns with HALA’s multi-layer human learning model. citeturn11search0 |
| *An Evaluation of Khanmigo…* citeturn11search1turn11search33 | entity["people","S. Shetye","researcher"] | Evaluates Khanmigo as CALL app; contributes early evidence base and limitations. citeturn11search1 | Useful as a case study: HALA can critique and extend—moving from content tutoring toward transformational and trust-governed partnership. citeturn11search1 |
| *ChatGPT-4 in evaluating open-ended exam responses* citeturn11search6 | entity["people","J.S. Jauhiainen","researcher"] et al. | Examines GenAI for assessment and feedback in higher education contexts. citeturn11search6 | Integration point for HALA Trust Infrastructure: assessment agents require transparency, bias controls, and relational governance (student consent, contestability). citeturn11search6 |
| *UNESCO Guidance for generative AI in education and research* citeturn11search3turn11search24 | entity["people","Fengchun Miao","education policy expert"] and entity["people","Wayne Holmes","education researcher"] | Global guidance emphasizing human-centered policy, capacity building, and guardrails. citeturn11search3 | Directly usable for HALA’s trust/governance layer; offers legitimacy and policy framing. citeturn11search3 |
| *U.S. ED AI report and guidance* citeturn16search5turn16search9 | entity["organization","Office of Educational Technology","us education office"] | Federal framing on AI in teaching/learning ecosystems and responsible use principles. citeturn16search5turn16search9 | Provides policy alignment and dissemination pathway for HALA in U.S. education contexts. citeturn16search5turn16search9 |

**Emerging Paradigms:**
- **LLMs as components inside pedagogically grounded ITS**: strongest evidence tends to come from systems that embed best practices (scaffolding, feedback loops), not generic chat interfaces. citeturn11search13turn11search0turn11search2  
- **Assessment and feedback automation**: GenAI is increasingly used for grading and feedback, raising trust, fairness, and transparency stakes. citeturn11search6turn11search3  
- **Ethics and governance moving into formal policy**: UNESCO and U.S. ED guidance signal that “responsible use” is increasingly codified rather than optional. citeturn11search3turn16search9  

**Identified Gaps:**
- **Transformational outcomes are rarely the explicit dependent variable**: most studies measure performance and engagement; fewer measure identity shift, meaning change, or long-term autonomy—the outcomes HALA targets. citeturn11search27turn19search0  
- **Somatic learning is nearly absent in AI tutor design**: even advanced frameworks emphasize Socratic dialogue and cognitive scaffolding more than embodied regulation, stress, or somatic sense-making. citeturn11search2turn15search3  
- **Governance is often policy-level, not interaction-level**: guidance exists, but many tutoring systems still lack “relationship contracts” (how memory is used, how errors are repaired, how agency boundaries are enforced). citeturn11search3turn4search10  

**Integration Opportunities:**
- **HALA as a design layer over tutoring architectures**: extend modular ITS frameworks by adding Trust Infrastructure artifacts (consent, transparency, contestability) and “human layers” (emotion, somatic signals, social belonging) as explicit design requirements. citeturn11search2turn5search11turn11search3  
- **Develop “agent pedagogy patterns” for education**: transform high-level principles (observable process, scaffolded autonomy, role separation) into reusable design patterns and evaluation rubrics. citeturn4search10turn12search3  
- **Bridge to policy and ethics**: align HALA Trust Infrastructure with UNESCO/U.S. ED principles so the framework is simultaneously research-grounded and deployment-ready. citeturn11search3turn16search9  

## Cross-domain synthesis for HALA

**White Spaces HALA uniquely addresses**
1. **Relationship governance as a first-class learning architecture**: Across agents, teaming, and education, governance is usually bolted on as “policy” or “safety,” while HALA treats it as layered infrastructure shaping both learning and trust (a gap visible from constitutional AI and enterprise adoption evidence). citeturn6search0turn10view0turn4search10  
2. **Whole-person learning applied to human–AI partnerships**: Education and HCI increasingly acknowledge emotion/wellbeing, but few frameworks integrate emotional + somatic + social dimensions *together* with agent capability progression. citeturn5search11turn15search3turn4search26  
3. **Co-development ladders (human + agent) with measurable stages**: Agent evaluation surveys highlight fragmented metrics; HALA can unify “developmental staging” across capabilities, autonomy, and relationship maturity. citeturn12search3turn4search17  
4. **Social learning for agents framed through social learning theory**: Multi-agent systems show emergent conventions and biases, but rarely translate to a Bandura-style learning lens (attention/retention/motivation/identity). HALA can operationalize this mapping. citeturn13search3turn18search0  

**Competing frameworks and differentiation**
- **Agent frameworks (AutoGen/CAMEL/ChatDev/MetaGPT)** compete on orchestration and multi-agent workflow but do not generally provide a human developmental ladder or trust-layer governance as a coherent theory of learning partnership. citeturn2search24turn2search25turn2search34turn2search31  
- **Alignment frameworks (Constitutional AI, SALMON, DPO)** provide value/behavior constraints but typically treat “values” as principles or preferences rather than relationship contracts and developmental co-evolution. citeturn6search0turn7search0turn6search1  
- **Human–AI teaming frameworks (guidelines, trust calibration reviews, complementarity models)** provide interaction principles but often under-specify affective/somatic layers and do not integrate persistent agent learning modalities (memory/tool/constitution) as part of a unified architecture. citeturn4search10turn4search26turn4search7turn2search9  
- **Instructional design and edtech frameworks (WILD, generative ITS)** cover wellbeing or tutoring design but usually stop short of treating the AI as a co-evolving agent with identity, memory governance, and team-like relational dynamics. citeturn5search11turn11search2turn11search13  

**Validation opportunities that would most strengthen HALA**
1. **Longitudinal co-learning studies (3–12 months)** where a human–agent pair is instrumented to measure: trust calibration, memory reliability, skill transfer, and human developmental outcomes (agency, emotional regulation, social belonging), not only task success. Anchor metrics in agent evaluation taxonomies. citeturn12search3turn4search26turn15search5  
2. **Randomized field experiments on “trust infrastructure” interventions**: compare teams using (a) standard copilots, (b) agentic memory/tools without governance, (c) HALA trust-layer governance (memory contracts, repair protocols, observable process), measuring errors, adoption, and performance. Enterprise evidence suggests integration + learning are the bottlenecks, so this is high-leverage. citeturn10view0turn9search25turn4search10  
3. **Multi-agent society stress tests**: evaluate whether HALA governance layers (reputation + role separation + sandboxing) reduce emergent bias and tragedy-of-commons patterns in agent collectives. citeturn13search3turn3search2  
4. **Educational RCTs measuring transformation variables**: build on AI tutor RCT designs but include transformative learning endpoints (frame shifts, identity narratives, autonomy) and emotional/somatic measures. citeturn11search13turn19search0turn15search5  

**Partnership targets**
- entity["organization","MIT Sloan School of Management","cambridge, ma, us"] / MIT NANDA researchers for enterprise learning-gap framing and field access. citeturn10view1turn4search19  
- entity["organization","Stanford HAI","stanford, ca, us"] for human-centered AI, evaluation, and collaboration research bridges. citeturn11search21turn6search32  
- entity["organization","Princeton HCI Group","princeton, nj, us"] for explainability-as-work-practice and collaboration design evidence. citeturn4search18  
- entity["organization","Khan Academy","education nonprofit"] research/practice teams for large-scale tutoring deployment contexts where trust + pedagogy constraints are real. citeturn11search37turn11search30  
- entity["company","Anthropic","ai safety company"] for constitutional methods and principled governance mechanisms that HALA can translate to “relationship constitutions.” citeturn6search0turn6search4  
- entity["company","Microsoft Research","ai research lab"] for multi-agent orchestration platforms as experimental substrate (AutoGen) and workplace adoption signals. citeturn2search24turn9search25  
- City St George’s / ITU Copenhagen groups for emergent convention/bias dynamics in LLM populations (a natural testbed for HALA trust-layer claims). citeturn13search3turn13search9  

**Publication strategy**
- **Core AI venues (agent learning, multi-agent, evaluation):** NeurIPS / ICML / ICLR / ACL (agent benchmarks, tool-use, multi-agent debate, alignment). citeturn6search5turn12search13turn3search24  
- **Human-centered venues (teaming, trust, pedagogy):** CHI / CSCW / AAMAS (human–agent teams, transparency, sustainability). citeturn4search10turn4search17  
- **Education-specific venues:** AIED / EDM / LAK / Computers & Education: Artificial Intelligence (generative ITS, learning analytics, applied studies). citeturn11search0turn11search27turn5search0  
- **Cross-sector/policy legitimacy:** UNESCO- and education-policy adjacent dissemination for Trust Infrastructure alignment (especially if HALA targets institutional adoption). citeturn11search3turn16search9  

**Trend alignment with HALA’s principles**
- HALA’s **M2/M3 emphasis (memory + tools)** aligns strongly with current agent SOTA: memory management and tool-use pipelines are central trends. citeturn2search9turn12search2turn1search0  
- HALA’s **trust-layer governance** aligns with the enterprise “learning gap” diagnosis and the rise of policy guidance for education; it diverges from many technical papers that still treat governance as secondary. citeturn10view0turn11search3turn4search33  
- HALA’s **whole-person learning** diverges from mainstream agent engineering (still dominated by declarative/procedural capability) but aligns with emerging wellbeing/emotion/somatic scholarship that remains under-incorporated into AI systems. citeturn5search11turn15search3turn4search26  

**Terminology mapping (HALA ↔ academic vocabulary)**
- “Dual Ladder” ↔ **developmental trajectories**, **capability maturity**, **scaffolding / ZPD**, **stage-by-stage formation** (cultural-historical activity theory). citeturn17search24turn17search32turn17search33  
- “Trust Infrastructure” ↔ **appropriate trust**, **trust calibration**, **trustworthy AI**, **governance frameworks**, **reputation systems**, **policy guardrails**. citeturn4search33turn3search2turn11search3  
- “Agent Learning Modalities” ↔ **in-context learning / retrieval**, **memory-augmented agents**, **tool use**, **constitutional/principle-based alignment**. citeturn2search9turn12search2turn6search0  
- “Agent pedagogy” ↔ **instructional scaffolding**, **Socratic tutoring**, **metacognitive prompting / reflection**, **mixed-initiative interaction**, **collaborative decision-making**. citeturn11search2turn6search3turn4search1  

**Citation foundation: 20 “must-cite” works to establish HALA credibility (cross-domain)**
- *Constitutional AI* (Bai et al.) citeturn6search0  
- *DPO* (Rafailov et al.) citeturn6search1  
- *SALMON* (Sun et al.) citeturn7search0  
- *Self-Rewarding Language Models* (Yuan et al.) citeturn7search1  
- *Scaling laws for reward overoptimization* (Rafailov et al.) citeturn7search7  
- *TrustAgent* (Hua et al.) citeturn3search10  
- *Voyager* (Wang et al.) citeturn1search0  
- *Reflexion* (Shinn et al.) citeturn1search1  
- *Toolformer* (Schick et al.) citeturn2search6  
- *ToolLLM/ToolBench* (Qin et al.) citeturn12search2  
- *MemGPT* (Packer et al.) citeturn2search9  
- *SWE-bench* (Jimenez et al.) citeturn12search0  
- *Agent evaluation survey* (Mohammadi et al.) citeturn12search3  
- *LLM multi-agent survey* (Guo et al.) citeturn3search19  
- *AutoGen* (Wu et al.) citeturn2search24  
- *Emergent social conventions and collective bias* (Ashery et al.) citeturn13search3  
- *Complementarity in human–AI collaboration* (Hemmer et al.) citeturn4search7  
- *When combinations of humans and AI are useful* (Vaccaro et al.) citeturn4search19  
- *Guidelines for Human–AI Interaction* (Amershi et al.) citeturn4search10  
- *UNESCO Guidance for generative AI in education and research* (Miao & Holmes) citeturn11search3  
- (High-impact adoption framing) *The GenAI Divide: State of AI in Business 2025* (Challapally et al.) citeturn10view1  

**Where HALA can leverage its Russian theoretical roots most credibly**
- HALA can position itself as a modern synthesis of cultural-historical developmental ideas—Vygotsky’s mediated learning and ZPD, Galperin’s planned stage-by-stage formation, and Davydov’s developmental instruction—translated into human–AI partnership design, where “mediation” is partly performed by agentic systems with memory/tools/constitutions. citeturn17search24turn17search32turn17search33  
- This is also a strategic differentiation against Western-centric framings that center individual cognition or usability guidelines without a developmental theory of internalization, staged autonomy, and social mediation as the engine of transformation. citeturn4search10turn19search0turn17search24