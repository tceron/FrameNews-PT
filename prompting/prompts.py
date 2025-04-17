import ast
import pdb
import pandas as pd
from collections import defaultdict

ALL_PROMPTS = {
    'zero1': """### PROMPT:
"{content}"

### TASK:
Classify the PROMPT above into exactly ONE of these frame categories:

"1" Economic
"2" Capacity and resources
"3" Morality
"4" Fairness and equality
"5" Legality, Constitutionality, Jurisdiction
"6" Crime and punishment
"7" Security and defense
"8" Health and safety
"9" Quality of life
"10" Cultural identity
"11" Public opinion
"12" Political
"13" Policy prescription and evaluation
"14" External regulation and reputation
"15" Other

### CLASSIFICATION GUIDELINES:

{guidelines}

Base your answer only on the PROMPT and the guidelines provided above. Answer as a single number ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15") corresponding to the most appropriate category.

### ANSWER:""",

    'zero2':"""
### CLASSIFICATION GUIDELINES: 

{guidelines}

### PROMPT:
"{content}"

### TASK:
Classify the PROMPT above into exactly ONE of the categories below. 
"1" Economic
"2" Capacity and resources
"3" Morality
"4" Fairness and equality
"5" Legality, Constitutionality, Jurisdiction
"6" Crime and punishment
"7" Security and defense
"8" Health and safety
"9" Quality of life
"10" Cultural identity
"11" Public opinion
"12" Political
"13" Policy prescription and evaluation
"14" External regulation and reputation
"15" Other

 Answer as a single number ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15") corresponding to the most appropriate category. 
 ### ANSWER:""",

   'zero3_noex':"""
### CLASSIFICATION GUIDELINES: 

FRAMES are the main gist of the article, the frame that is the most likely to stick in
the readers’ mind. It is assigned after reading the whole article and absorbing the main
takeaway. A helpful test in determining the primary frame is considering how you would
describe the article to a friend. If there are many frames in an article that seem to be
equally primary, the annotators will default to using the headline frame as the primary
frame.

Categories of frames: 

1. Economic:
The costs, benefits, or any monetary/financial implications of the issue (to an individual, family,
organization, community or to the economy as a whole).
Can include the effect of policy issues on trade, markets, wages, employment or unemployment,
viability of specific industries or businesses, implications of taxes or tax breaks, financial
incentives, etc.

2. Capacity and resources:
The lack or availability of resources (time, physical, geographical, space, human, and financial
resources). The capacity of existing systems and resources to carry out policy goals.
The easiest way to think about it is in terms of there being "not enough" or “enough” of
something. The capacity or resources may be an impediment to solving a problem or adequately
addressing an issue.

3. Morality:
Any perspective that is compelled by religious doctrine or interpretation, duty, honor,
righteousness or any other sense of ethics or social or personal responsibility.
It is sometimes presented from a religious perspective (i.e. “eye for an eye”), but non-religious
frames can also be used. For example, the moral imperatives to help others can be used to justify military intervention or foreign aid, social programs such as Medicare, welfare, and food stamps.
Appeals that a policy move “is just the right thing to do” or “would indicate a recognition of our
shared humanity” may reflect humanist morality. The commitment aspect of marriage would
evoke feelings of morality. Environmental arguments that focus on responsible stewardship or
“leaving something for our children” are based in a sense of responsibility or morality.
Lawbreakers, including illegal immigrants, can be presented as fundamentally immoral,
conversely breaking a law that is bad or unjust can be presented as moral (e.g., Rosa Parks).
Enacting protective legislation, such as laws that protect children from pedophiles, guns,
violence, poverty, or failure to do so can also be presented using moral frames.

4. Fairness and equality:
The fairness, equality or inequality with which laws, punishment, rewards, and resources are
applied or distributed among individuals or groups. Also the balance between the rights or
interests of one individual or group compared to another individual or group.
Fairness and Equality frame signals often focus on whether society and its laws are equally
distributed and enforced across regions, race, gender, economic class, etc. Many gender and race
issues, in particular, include equal pay, access to resources such as education, healthcare and
housing. Another example could be fairness considerations about whether punishments are
proportional to crimes committed. The frame is also used when discussing social justice,
discrimination and talk of an inmate’s innocence or exogeneration.

5. Legality, Constitutionality, Jurisdiction:
The legal, constitutional, or jurisdictional aspects of an issue. Legal aspects include existing
laws, reasoning on fundamental rights and court cases; constitutional aspects include all
discussion of constitutional interpretation and/or potential revisions; jurisdiction includes any
discussion of which government body should be in charge of a policy decision and/or the
appropriate scope of a body’s policy reach. This frame deals specifically with the authority of
government to regulate, and the authority of individuals/corporations to act independently of
government.
Of special note are constraints imposed on freedoms granted to individuals, government, and
corporations via the Constitution, Bill of Rights and other amendments. Some frequent
arguments and issues are: i) the right to bear arms; ii) equal protection; iii) free speech and
expression; iv) the constitutionality of restricting individual freedoms and imposing taxes; v)
conflicts between state, local or federal regulation and authority, or between different branches of
government; vi) legal documentation (green card, visas, passports, driver licenses, marriage
license, etc.).

6. Crime and punishment:
The violation of policies and its consequences. It includes enforcement and interpretation of civil
and criminal laws, sentencing and punishment with retribution or sanctions.
This frame includes: i) deportation when an individual does not have the necessary documents
that grant legal standing; ii) increases or reductions in crime; iii) punishment and execution; iv)
resources analysis like DNA analysis. Usually found together with other frames, such as
Economic, Legality, constitutionality and jurisdiction, Morality, and Capacity and resources. The
primary frame should be chosen according to where the emphasis is.

7. Security and defense:
Any threat to a person, group, or nation, or any defense that needs to be taken to avoid that
threat.
Security and Defense frames differ from Health and Safety frames in that Security and Defense
frames address a preemptive action to stop a threat from occurring, whereas Health and Safety
frames address steps taken to ensure safety in the event that something happens. It can include
efforts to build a border fence or “secure the borders,” issues of national security including
resource security, efforts of individuals to secure homes, neighborhoods or schools, and efforts
such as guards and metal detectors that would defend children from a possible threat. Discussion
regarding terrorist activity should be coded as Security and Defense (e.g. arrests of terrorists,
immigrants linked to terrorism activity, increased border security to prevent terrorism). Arrests at
the border will receive both a Crime and Punishment and Security and Defense frame but the
primary frame would be Security and Defense since the action is taking place on the border. All
terrorist attacks are coded as Security and Defense, but attention should be paid to potential
criminal, legal, or any other aspects and double coded accordingly.

8. Health and safety:
The potential health and safety outcomes of any policy issue (e.g. health care access and
effectiveness, illness, disease, sanitation, carnage, obesity, mental health infrastructure and
building safety). Also policies taken to ensure safety in case of a tragedy would fit under this
(e.g. emergency preparedness kits, lock down training in schools, disaster awareness classes for
teachers).
It includes any discussion of the various capital punishment methods and procedures and ##any
mentions of refugees##. Often used in conjunction with Quality of Life.

9. Quality of life:
The benefits and costs of any policy on quality of life.
The effects of a policy on people’s wealth, mobility, access to resources, happiness, social
structures, ease of day-to-day routines, quality of community life, etc. It includes any mention of
people receiving generic “benefits”, adoptions, and weddings. Often used in conjunction with
Health and Safety.

10. Cultural identity:
The social norms, trends, values and customs constituting any culture(s).
It includes: i) language issues and language learning; ii) patriotism and national traditions, the
history of an issue or the significance of an issue within a group or subculture; iii) census and
demographics; iv) cultural shifts in a group or society; v) cultural norms of ethnic and political
groups. May also include stereotypes or assumed preferences and reactions of a group (e.g., an
affinity for Republicans to wear cowboy hats); vi) references and quotations of famous people
like politicians, leaders or representatives of a subculture.

11. Public opinion:
The opinion of the general public.
It includes references to general social attitudes, protests, polling and demographic information,
as well as any public passage of a proposition or law (i.e. “California voters passed Prop 8”). All
the opinions that represent the sentiment of a group will be coded as Public opinion. However, a
group of experts in a particular domain gets coded according to their domain (e.g. police officers
in Crime and Punishment, or climate scientists in Capacity and Resources).

12. Political:
In general, any political considerations surrounding an issue.
It includes political actions, maneuvering, efforts or stances towards an issue (e.g. partisan
filibusters, lobbyist involvement, deal-making and vote trading), mentions of political entities or
parties (e.g., Democrats, Republicans, Libertarians, Green Party). When a headline mentions
“both sides” this refers to politics.

13. Policy prescription and evaluation:
The analysis of whether hypothetical policies will work or existing policies are effective. What
is/isn’t currently allowed and what should/shouldn’t be done? “Policy” encompasses formal
government regulation (e.g., federal or state laws) as well as regulation by businesses (e.g.,
sports arenas not allowing the sale of alcohol).
This frame dimension—perhaps more than any other—is likely to appear frequently across texts.
Yet care should be given to only use this code category as the primary frame when the main
thrust of an article is really about policy, for example when it describes the success and failure of
existing policies or proposes policy solutions to a problem.

14. External regulation and reputation:
In general, the country’s external relations with another nation; the external relations of
a state with another.
This frame includes: i) trade agreements and outcomes; ii) comparisons of policy outcomes
between different groups or regions; iii) the perception of one nation, state, and/or group by
another (for example, international criticisms of the United States maintaining capital
punishment); iv) border relations, interstate or international efforts to achieve policy goals; v)
alliances or disputes between groups.

15. Other:
Any frame signal that does not fit in the first 14 dimensions.

### PROMPT:
"{content}"

### TASK:
Classify the PROMPT above into exactly ONE of the categories below. 
"1" Economic
"2" Capacity and resources
"3" Morality
"4" Fairness and equality
"5" Legality, Constitutionality, Jurisdiction
"6" Crime and punishment
"7" Security and defense
"8" Health and safety
"9" Quality of life
"10" Cultural identity
"11" Public opinion
"12" Political
"13" Policy prescription and evaluation
"14" External regulation and reputation
"15" Other

 Answer as a single number ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15") corresponding to the most appropriate category. 
 ### ANSWER:"""
            }
