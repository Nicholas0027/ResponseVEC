# CES 2020 preliminary target review

Generated evidence for manual validity review. Inclusion decisions must
be based on wording/branching validity, never observed model performance.

## CC20_433a

- Page: 41
- Coverage: 1.0000; categories: 4
- Automated flags: branch_risk
- Manual decision: INCLUDE
- Construct: party identification
- Reason: fixed text and high coverage; primary in-construct identity target

Questionnaire 40 Page: implicit_page_CC20_433a CC20_433a- required SINGLE CHOICE Generally speaking, do you think of yourself as a ... ? varlabel 3 pt party ID required HARD 1 ○ Democrat 2 ○ Republican 3 ○ Independent 4 ○ Other (open [CC20_433_t]) 8 Skipped 9 Not Asked Page: implicit_page_CC20_433_dem CC20_433_dem- Show if CC20_433a==1 SINGLE CHOICE Would you call yourself a strong Democrat or not so strong Democrat?

## CC20_420_6

- Page: 35
- Coverage: 1.0000; categories: 2
- Automated flags: branch_risk, dynamic_text
- Manual decision: INCLUDE
- Construct: foreign policy / multilateral military action
- Reason: fixed policy meaning; binary punch text manually recovered from parent block

varlabel Race/ethnicity politicians - $HouseCand2Name - open Page: implicit_page_CC20_416c_other CC20_416c_other- Show if CC20_416c == 5 and CurrentHouseName OPEN TEXTBOX What race or ethnicity is $CurrentHouseName? varlabel Race/ethnicity politicians - $CurrentHouseName - open Page: implicit_page_CC20_420 We'd now like to ask you about some issues facing the country.

## CC20_430a_8

- Page: 37
- Coverage: 1.0000; categories: 2
- Automated flags: multiple_select
- Manual decision: EXCLUDE
- Construct: political participation
- Reason: implicit multi-select punch with ambiguous item-level wording in PDF

please check the 'not sure' box. min 0 max 100 dk 1 left All from sales tax right All from income taxes show_value 1 Page: implicit_page_CC20_430a CC20_430a MULTIPLE CHOICE During the past year did you ... (Check all that apply) varlabel Past year 1 □ Attend local political meetings (such as school board or city council) 2 □ Put up a political sign (such as a lawn sign or bumper sticker) 3 □ Work for a candidate or campaign 4 □ Attend a political protest, march or demonstration

## CC20_443_4

- Page: 45
- Coverage: 0.9999; categories: 5
- Automated flags: none
- Manual decision: INCLUDE
- Construct: state spending / law enforcement
- Reason: fixed ordinal policy target

CC20_443- prompt once on skip GRID State legislatures must make choices when making spending decisions on important state programs. How would you like your legislature to spend money on each of the five areas below? varlabel State legislature spending required SOFT collapsible 0 ROWS CC20_443_1 Welfare CC20_443_2 Health Care CC20_443_3 Education CC20_443_4 Law Enforcement CC20_443_5 Transportation/Infrastructure COLUMNS 1 ○ Greatly increase 2 ○ Slightly increase 3 ○ Maintain 4 ○ Slightly decrease 5 ○ Greatly decrease 8 Skipped 9 Not Asked

## CC20_442d

- Page: 44
- Coverage: 0.9998; categories: 2
- Automated flags: none
- Manual decision: INCLUDE
- Construct: immigration and asylum policy
- Reason: fixed binary policy target

required SOFT collapsible 0 ROWS CC20_442a Assassination of Iranian General Qasem Soleimani CC20_442b Withdraw the United States from the Iran Nuclear Accord and reimpose sanctions on Iran CC20_442c Declare a national emergency to permit construction of border wall with Mexico CC20_442d Suspend a program that allows migrants to remain in the US while their asylum cases were being decided. CC20_442e Withdraw troops from Kurdish-controlled region of northern Syria on the border with Turkey COLUMNS 1 ○ Support 2 ○ Oppose 8 Skipped 9 Not Asked

## CC20_440a

- Page: 42
- Coverage: 0.9994; categories: 5
- Automated flags: none
- Manual decision: INCLUDE
- Construct: racial privilege attitudes
- Reason: fixed ordinal attitude target

8 ○ Not sure 98 Skipped 99 Not Asked Page: implicit_page_CC20_440_grid CC20_440_grid DYNAMIC GRID Do you agree or disagree with the following statements? varlabel Racial/Sexual agreement ROWS CC20_440a White people in the U.S. have certain advantages because of the color of their skin. CC20_440b Racial problems in the U.S. are rare, isolated situations. CC20_440c Women seek to gain power by getting control over men. CC20_440d Women are too easily offended. COLUMNS 1 ○ Strongly agree 2 ○ Somewhat agree 3 ○ Neither agree nor disagree 4 ○ Somewhat disagree 5 ○ Strongly disagree

## CC20_441a

- Page: 43
- Coverage: 0.9993; categories: 5
- Automated flags: branch_risk
- Manual decision: INCLUDE
- Construct: racial resentment
- Reason: fixed ordinal attitude target; race-conditional follow-ups are not this item

Page: implicit_page_CC20_441_grid CC20_441_grid- prompt once on skip GRID How much do you agree or disagree with the following statements? varlabel Racial Resentment required SOFT ROWS CC20_441a Irish, Italians, Jewish and many other minorities overcame prejudice and worked their way up. Blacks should do the same without any special favors. CC20_441b Generations of slavery and discrimination have created conditions that make it difficult for blacks to work their way out of the lower class. CC20_441e- Show if race != 1 I resent when Whites deny the existence of racial discrimination. CC20_441f- Show if race != 1 Whites get away with offenses that African Americans would never get away with. CC20_441g- Show if race != 1 Whites do not go to great lengths to understand the problems African Americans face. COLUMNS 1 ○ Strongly agree 2 ○ Somewhat agree

## CC20_431a

- Page: 39
- Coverage: 0.9986; categories: 2
- Automated flags: branch_risk, multiple_select
- Manual decision: INCLUDE
- Construct: campaign contact / political mobilization
- Reason: fixed binary behavioral target; downstream follow-up branching is irrelevant

Show if 6 in CC20_430a OPEN INTEGER TEXTBOX Approximately how much did you contribute to all candidates and committees over the last year? varlabel Amount contribute to all candidates/committees min 1 max 1000000 left &dollar; Page: implicit_page_CC20_431a CC20_431a SINGLE CHOICE Did a candidate or political campaign organization contact you during the 2020 election? varlabel Ever run for elective office 1 ○ Yes 2 ○ No 8 Skipped 9 Not Asked Page: implicit_page_CC20_431b CC20_431b- Show if CC20_431a == 1 MULTIPLE CHOICE How did these candidates or campaigns contact you? Check all that apply. varlabel Run for office 1 □ In person 2 □ Phone call

## CC20_432a

- Page: 40
- Coverage: 0.9971; categories: 2
- Automated flags: branch_risk, multiple_select
- Manual decision: INCLUDE
- Construct: political participation / candidacy
- Reason: fixed binary lifetime behavior; rare outcome retained as a hard target

Questionnaire 39 Page: implicit_page_CC20_432a CC20_432a SINGLE CHOICE Have you ever run for elective office at any level of government (local, state or federal)? varlabel Past year 1 ○ Yes 2 ○ No 8 Skipped 9 Not Asked Page: implicit_page_CC20_432b CC20_432b- Show if CC20_432a == 1 MULTIPLE CHOICE Which of the following offices have you run for? Select all that apply. varlabel Run for office 1 □ School Board 2 □ Other local board or commission (e.g.

## CC20_416a

- Page: 34
- Coverage: 0.9774; categories: 6
- Automated flags: branch_risk, dynamic_text
- Manual decision: EXCLUDE
- Construct: candidate race perception
- Reason: target question changes with district-specific candidate name and availability

HouseCand1Name or HouseCand2Name or CurrentHouseName GRID What is the race or ethnicity of the following candidates or politicians? varlabel Race/ethnicity politicians ROWS CC20_416a- Show if HouseCand1Name $HouseCand1Name CC20_416b- Show if HouseCand2Name $HouseCand2Name CC20_416c- Show if CurrentHouseName and (not HouseCand1IncumbentNum) and (not HouseCand2IncumbentNum) $CurrentHouseName COLUMNS 1 ○ White 2 ○ Black 3 ○ Hispanic
