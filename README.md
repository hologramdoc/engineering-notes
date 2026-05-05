# Engineering Notes Blog

Hugo + PaperMod 기반 기술 블로그.

## 로컬 개발 시작하기

### 1. 사전 설치

```bash
# Hugo 설치 (macOS)
brew install hugo

# 테마 설치
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
git submodule update --init --recursive
```

### 2. 로컬 서버 실행

```bash
hugo server -D
# http://localhost:1313 에서 확인
```

### 3. 새 포스트 작성

```bash
hugo new posts/my-new-post.md
```

---

## GitHub Pages 배포

### 1. GitHub 저장소 생성

```bash
git init
git add .
git commit -m "Initial blog setup"
gh repo create engineering-notes --public --push
```

### 2. GitHub Pages 설정

GitHub 저장소 → Settings → Pages → Source: **GitHub Actions**

### 3. hugo.toml 수정

```toml
baseURL = "https://your-github-username.github.io/engineering-notes/"
```

### 4. 커스텀 도메인 (선택)

```toml
baseURL = "https://your-domain.com/"
```

---

## 블로그 구조

```
content/
├── about.md                    # 포트폴리오/소개
└── posts/
    ├── tensorrt-fp16-int8-optimization.md
    ├── onnx-conversion-troubleshooting.md
    ├── mlops-prometheus-manufacturing.md
    ├── edge-ai-vision-inspection-architecture.md
    └── lcd-mura-demura-algorithm.md
```

---

## 커스터마이징

`hugo.toml`에서 수정:
- `baseURL`: 실제 배포 URL
- `params.author`: 이름
- `params.socialIcons`: GitHub, LinkedIn 등 링크
