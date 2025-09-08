# Makefile for LaTeX presentations
# Multiple slide decks for different audiences
MAIN = final_complete_presentation
WORKING = main_working
BEGINNER = beginner_slides_simple
SECTIONS = sections/*.tex
IMAGES = images/*.png images/*.pdf

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex
FLAGS = -interaction=nonstopmode -halt-on-error

# Default target - build final presentation
all: $(MAIN).pdf

# Build final complete presentation
$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(FLAGS) $(MAIN).tex
	$(LATEX) $(FLAGS) $(MAIN).tex

# Build working slides
$(WORKING).pdf: $(WORKING).tex $(SECTIONS)
	$(LATEX) $(FLAGS) $(WORKING).tex
	$(LATEX) $(FLAGS) $(WORKING).tex

# Build beginner slides
$(BEGINNER).pdf: $(BEGINNER).tex
	$(LATEX) $(FLAGS) $(BEGINNER).tex
	$(LATEX) $(FLAGS) $(BEGINNER).tex

# Build only final presentation
final: $(MAIN).pdf

# Build only working slides
working: $(WORKING).pdf

# Build only beginner slides
beginner: $(BEGINNER).pdf

# Full build with bibliography (if needed)
full: $(MAIN).tex $(WORKING).tex $(BEGINNER).tex
	$(LATEX) $(FLAGS) $(MAIN).tex
	$(BIBTEX) $(MAIN) || true
	$(LATEX) $(FLAGS) $(MAIN).tex
	$(LATEX) $(FLAGS) $(MAIN).tex
	$(LATEX) $(FLAGS) $(WORKING).tex
	$(LATEX) $(FLAGS) $(WORKING).tex
	$(LATEX) $(FLAGS) $(BEGINNER).tex
	$(LATEX) $(FLAGS) $(BEGINNER).tex

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.bbl *.blg *.synctex.gz

# Clean everything including PDFs
distclean: clean
	rm -f $(MAIN).pdf $(WORKING).pdf $(BEGINNER).pdf

# Watch for changes and rebuild
watch:
	while inotifywait -e modify $(MAIN).tex $(SECTIONS); do \
		make; \
	done

# Create placeholder logo if it doesn't exist
images/gnuradio_logo.png:
	mkdir -p images
	convert -size 200x200 xc:blue -fill white -gravity center -pointsize 40 -annotate +0+0 "GR" images/gnuradio_logo.png

# Help
help:
	@echo "Available targets:"
	@echo "  all       - Build final complete presentation (default)"
	@echo "  final     - Build final_complete_presentation.pdf"
	@echo "  working   - Build main_working.pdf"
	@echo "  beginner  - Build beginner_slides_simple.pdf"
	@echo "  full      - Full build with bibliography for all slides"
	@echo "  clean     - Remove auxiliary files"
	@echo "  distclean - Remove all generated files"
	@echo "  watch     - Auto-rebuild on file changes"
	@echo "  help      - Show this help message"

.PHONY: all final working beginner full clean distclean watch help