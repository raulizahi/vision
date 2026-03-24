CC        = clang
CFLAGS    = -Wall -Wextra -O2
OBJCFLAGS = -Wall -Wextra -O2 -fobjc-arc
FRAMEWORKS = -framework Foundation \
             -framework Vision     \
             -framework AppKit     \
             -framework CoreGraphics \
             -framework CoreText   \
             -framework ImageIO    \
             -framework UniformTypeIdentifiers

TARGET  = vision
OBJECTS = main.o face_detector.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(FRAMEWORKS)

main.o: main.c face_detector.h
	$(CC) $(CFLAGS) -c main.c

face_detector.o: face_detector.m face_detector.h
	$(CC) $(OBJCFLAGS) -c face_detector.m

clean:
	rm -f $(OBJECTS) $(TARGET)
