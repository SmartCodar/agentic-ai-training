# Python LLM Engineering Course 🚀

A comprehensive course on Large Language Models (LLMs) and AI Engineering with Python. From basics to advanced topics in 22 days.

## 📚 Course Overview

| Module | Topics | Duration | Prerequisites |
|--------|---------|-----------|---------------|
| Python Basics | Core Python, OOP, Testing | Days 1-9 | None |
| FastAPI | Web Development, APIs | Days 10-16 | Python Basics |
| LLM Engineering | Transformers, Prompting | Days 17-22 | All Previous |

## 📖 Course Content

### Module 1: Python Fundamentals
| Day | Topic | Description |
|-----|-------|-------------|
| 1 | [Variables & Types](articles/01-basics/01-Python-Basics-Variables-Types-Operators.md) | Python basics, data types |
| 2 | [Flow Control](articles/01-basics/02-Python-Flow-Control-Loops-Conditions.md) | Loops, conditions |
| 3 | [Functions](articles/01-basics/03-Python-Functions-Modular-Programming.md) | Function definitions |
| 4 | [Modules](articles/01-basics/04-Python-Modules-Packages.md) | Code organization |
| 5 | [OOP](articles/01-basics/05-Python-OOP.md) | Classes, inheritance |
| 6 | [File Handling](articles/01-basics/06-Python-File-Handling.md) | File operations |
| 7 | [Testing](articles/01-basics/07-Python-Testing-Debugging.md) | Unit tests, debugging |
| 8 | [Functional](articles/01-basics/08-Python-Functional-Programming.md) | Functional programming |
| 9 | [Project Setup](articles/01-basics/09-Python-Project-Setup.md) | Project structure |

### Module 2: FastAPI Development
| Day | Topic | Description |
|-----|-------|-------------|
| 10 | [Async](articles/02-fastapi/10-Python-Async-Programming.md) | Async programming |
| 11 | [Aiohttp](articles/02-fastapi/11-Python-Aiohttp-Client.md) | HTTP client |
| 12 | [FastAPI Intro](articles/02-fastapi/12-Python-FastAPI.md) | Framework basics |
| 13 | [Routes](articles/02-fastapi/13-Python-FastAPI-Routes.md) | API endpoints |
| 14 | [Pydantic](articles/02-fastapi/14-Python-Pydantic.md) | Data validation |
| 15 | [Project](articles/02-fastapi/15-Python-FastAPI-Project.md) | Full application |
| 16 | [Mini-Project](articles/02-fastapi/16-Python-FastAPI-Mini-Project.md) | Practice project |

### Module 3: LLM Engineering
| Day | Topic | Description |
|-----|-------|-------------|
| 17 | [Transformers](articles/03-llm/17-Python-LLM-Transformers.md) | Architecture, concepts |
| 18 | [Applications](articles/03-llm/18-Python-LLM-Applications.md) | Use cases |
| 19 | [Prompt Engineering](articles/03-llm/19-Python-Prompt-Engineering.md) | Prompt design |
| 20 | [Role Prompting](articles/03-llm/20-Python-Role-Based-Prompting.md) | Specialized agents |
| 21 | [Chain-of-Thought](articles/03-llm/21-Python-Chain-Of-Thought.md) | Reasoning steps |
| 22 | [Evaluation](articles/03-llm/22-Python-Prompt-Evaluation.md) | Quality metrics |

## 🗂️ Repository Structure

```
python-llm-course/
├── articles/                  # Course content
│   ├── 01-basics/            # Days 1-9
│   ├── 02-fastapi/           # Days 10-16
│   └── 03-llm/               # Days 17-22
│
├── code/                      # Code examples
│   ├── basics/               # Python fundamentals
│   ├── fastapi/              # FastAPI projects
│   └── llm/                  # LLM implementations
│       ├── transformers/     # Transformer examples
│       ├── prompts/          # Prompt engineering
│       ├── agents/           # Role-based agents
│       └── evaluation/       # Testing frameworks
│
├── exercises/                 # Practice problems
│   ├── basic/               
│   ├── intermediate/        
│   └── advanced/           
│
├── projects/                  # Complete projects
│   ├── chatbot/             
│   ├── code-assistant/      
│   └── content-generator/   
│
└── resources/                # Additional materials
    ├── images/              
    ├── notebooks/          
    └── references/         
```

## ⚡ Quick Start

1. **Setup Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. **Choose Your Path**
- 🔰 **Beginner**: Start with Module 1 (Days 1-9)
- 🌐 **Web Developer**: Jump to Module 2 (Days 10-16)
- 🤖 **AI Engineer**: Focus on Module 3 (Days 17-22)

3. **Learning Tips**
- Complete exercises in each article
- Build the projects
- Practice with code examples

## 🛠️ Technical Requirements

### Python Environment
- Python 3.7+
- Virtual environment (venv or conda)
- IDE with Python support (VS Code recommended)

### Key Dependencies
```txt
# Core
numpy>=1.21.0
pandas>=1.3.0
pydantic>=1.8.0

# Web Development
fastapi>=0.68.0
uvicorn>=0.15.0
aiohttp>=3.8.0

# LLM & AI
transformers>=4.20.0
openai>=0.27.0
nltk>=3.6.0
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/python-llm-course.git
cd python-llm-course
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start learning:
- Begin with Day 1 in articles/01-basics/
- Follow the course sequence
- Complete exercises as you go
- Build projects to apply knowledge

## 📖 Course Navigation

### Article Structure
Each article includes:
- Overview and objectives
- Prerequisites
- Technical requirements
- Time estimates
- Content with examples
- Exercises and projects
- Best practices
- Navigation links

### Code Examples
- Found in the code/ directory
- Organized by topic
- Fully documented
- Tested and production-ready

### Exercises
Three difficulty levels:
- Basic: Core concepts
- Intermediate: Combined skills
- Advanced: Real-world problems

## 🤝 Contributing

We welcome contributions:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Hugging Face for Transformers
- FastAPI team
- Python community

## 📬 Contact

- GitHub Issues for bugs and features
- Discussions for questions
- Pull Requests for contributions

Happy Learning! 🚀
