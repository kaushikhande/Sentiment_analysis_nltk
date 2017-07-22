\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pictreport}
% Passes and class options to the underlying article class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions
\LoadClass[a4paper,12pt]{book}
\RequirePackage{graphicx}
%\RequirePackage{minted}
\RequirePackage{float}
\RequirePackage[a4paper,bindingoffset=0.2in,left=1.5in,right=1in,top=1in,bottom=1.25in,footskip=.25in]{geometry}
\RequirePackage{mathptmx}
\RequirePackage{anyfontsize}
\RequirePackage{setspace}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{lipsum}
\RequirePackage{mfirstuc}
%\RequirePackage{showframe}

\RequirePackage{tikz}
\usetikzlibrary{arrows,shapes,snakes,automata,backgrounds,petri}


%change the section heading and font size
\titleformat{\section}
  {\normalfont\fontsize{12}{15}\bfseries\uppercase}{\thesection}{1em}{}
 
 \titleformat{\subsection}
  {\normalfont\fontsize{12}{15}\bfseries}{\thesubsection}{1em}{}
  
 \titleformat{\subsubsection}
  {\normalfont\fontsize{12}{15}}{\thesubsubsection}{1em}{}
  
  
%paragraph double line space and indentation of the paragraph  
\linespread{1.5}
\setlength{\parindent}{11mm}

%index page formatting
\renewcommand*\contentsname{\bfseries{\centerline{INDEX}}}



%to correct the image surroung blank space
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.15}
\renewcommand{\floatpagefraction}{0.8}
\renewcommand{\textfraction}{0.1}
\setlength{\floatsep}{5pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{5pt plus 2pt minus 2pt}
\setlength{\intextsep}{5pt plus 2pt minus 2pt}



\newcommand{\picttitlepage}[4] {
\setlength{\parindent}{0mm}
\begin{center}
{\bfseries A DISSERTATION REPORT ON \\}
 \vspace*{2\baselineskip}
{\bfseries \fontsize{16}{12} \selectfont #1 \\ \vspace*{2\baselineskip}}
{\fontsize{12}{12} \selectfont SUBMITTED TO THE SAVITRIBAI PHULE PUNE UNIVERSITY,
PUNE \\ IN PARTIAL FULFILLMENT OF THE REQUIREMENTS \\
FOR THE AWARD OF THE DEGREE OF\\
\vspace*{2\baselineskip}}
{\bfseries \fontsize{14}{12} \selectfont MASTER OF ENGINEERING (Computer
Engineering) \\
\vspace*{2\baselineskip}} 
{\bfseries \fontsize{14}{12} \selectfont BY \\ 
\vspace*{2\baselineskip}} 
#2 \hspace{25 mm} Exam No: #3 \\
\vspace*{1\baselineskip}
{\bfseries \fontsize{14}{12} \selectfont Under The Guidance of\\  
\vspace*{1\baselineskip}} 
 #4\\
\includegraphics[width=100pt]{pict.png} \\
{\bfseries \fontsize{14}{12} \selectfont DEPARTMENT OF COMPUTER ENGINEERING \\
PUNE INSTITUTE OF COMPUTER TECHNOLOGY \\
Sr.  No. 27, Pune Satara Road, Dhankawadi  \\
Pune 411043 \vspace*{2\baselineskip}}
\end{center}}

\newcommand{\pictcertificate}[4] {
\begin{figure}[ht]
\centering
\includegraphics[width=100pt]{pict.png}
\end{figure}

{\bfseries \fontsize{14}{12} \selectfont \centerline{PUNE INSTITUTE OF COMPUTER TECHNOLOGY}
\centerline{DEPARTMENT OF COMPUTER ENGINEERING}
\vspace*{1\baselineskip}} 


{\bfseries \fontsize{16}{12} \selectfont \centerline{CERTIFICATE} 
\vspace*{1\baselineskip}} 

\centerline{This is to certify that the dissertation entitled}
\vspace*{1\baselineskip} 

\begin{center}
{\bfseries \fontsize{14}{12} \selectfont {``#1''} 
\vspace*{1\baselineskip}}
\end{center}

\centerline{Submitted by}
\vspace*{1\baselineskip} 

\centerline{ #2 \hspace{25 mm}  Exam No: #3}
\vspace*{1\baselineskip} 
is a bonafide work carried out by him under the supervision of  #4 and it
is submitted towards the partial fulfillment of the requirement of Savitribai Phule Pune University, for the award of the degree of Master of Engineering (Computer Engineering) 
\vspace*{3\baselineskip} 


\bgroup
\def\arraystretch{0.7}
\begin{tabular}{c c c}
 #4 &  Dr. R. B. Ingle &  Dr. P. T. Kulkarni \\								
Internal Guide   &  Head &  Principal\\
Dept. of Computer Engg.  &	Dept. of Computer Engg. &  \\
PICT, Pune-43 &  PICT, Pune-43 & PICT, Pune-43\\
\end{tabular}
}

\newcommand{\companycertificate}[5] {
	\newpage
\vspace*{4\baselineskip}

{\bfseries \fontsize{16}{16} \selectfont \centerline{CERTIFICATE}  
\vspace*{3\baselineskip}}
\centerline{This is to certify that the dissertation report
entitled}
\vspace*{1\baselineskip}
\begin{center}
{\bfseries \fontsize{12}{12} \selectfont {``#1''}
\vspace*{1\baselineskip}}
\end{center}
\centerline{Submitted by}
\vspace*{1\baselineskip}
\centerline{#2 \hspace{25 mm} Exam No: #3}
\vspace*{1\baselineskip}
is a bonafide work carried out by him with the Sponsorship from Webonise Lab Pune, under the supervision of #4.
\vspace*{3\baselineskip}

#5 \\				
Co-Founder and VP - Engineering \\
Name and Sign External Guide 				
\vspace*{2\baselineskip}

Place : Pune\\
Date : 
}

\newcommand{\pictack}[2] {
{\bfseries \fontsize{14}{12} \selectfont \centerline{ACKNOWLEDGEMENT}
\vspace*{1\baselineskip}}

\textit{It gives me great pleasure in presenting the dissertation report for my
dissertation on {\bfseries \fontsize{12}{12} \selectfont ``#1''}.}
\vspace*{1.5\baselineskip}

 \textit{I would like to take this opportunity to thank my internal guide
 \textbf{Prof. A. G. Phakatkar} for giving me all the help and guidance I needed. I am  really grateful for her kind support throughout the analysis and design phase. Her valuable criticism and suggestions were very helpful.} \vspace*{1.5\baselineskip}

 \textit{I am grateful to \textbf{Dr. Rajesh B. Ingle}, Head of Computer
 Engineering Department, Pune Institute of Computer Technology for his indispensable  support, priceless suggestions and for most valuable time lent as and when required.
 I also thank \textbf{Dr. P. T. Kulkarni}, Principal, Pune Institute of Computer Technology for his encouragement and providing required resources.}
\vspace*{1.5\baselineskip}

\textit{In the end my special thanks to \textbf{Mr. Hrishikesh Kasar} for providing various resources such as well-equipped lab with all needed software platforms, continuous Internet connection, for my dissertation work.\\\\}
\vspace*{1\baselineskip}

Kaushik S. Hande\\
(M.E. Computer Engg.)
}



\newenvironment{pictabstract}
{  \newpage {\bfseries \fontsize{14}{12} \selectfont \centerline{ABSTRACT} 
\vspace*{2\baselineskip}} \setlength{\parindent}{11mm} }
{ \setlength{\parindent}{0mm} }

