# Copyright (c) 2026 Raul I. Lopez
# SPDX-License-Identifier: MIT
# See LICENSE file for details.

CC        = clang
CFLAGS    = -Wall -Wextra -O2
OBJCFLAGS = -Wall -Wextra -O2 -fobjc-arc
FRAMEWORKS = -framework Foundation \
             -framework Vision     \
             -framework AppKit     \
             -framework CoreGraphics \
             -framework CoreText   \
             -framework ImageIO    \
             -framework UniformTypeIdentifiers \
             -framework AVFoundation \
             -framework CoreMedia  \
             -framework CoreVideo

TARGET  = vision
OBJECTS = main.o face_detector.o video_detector.o body_detector.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(FRAMEWORKS)

main.o: main.c face_detector.h
	$(CC) $(CFLAGS) -c main.c

face_detector.o: face_detector.m face_detector.h face_detector_internal.h
	$(CC) $(OBJCFLAGS) -c face_detector.m

video_detector.o: video_detector.m video_detector.h face_detector.h face_detector_internal.h
	$(CC) $(OBJCFLAGS) -c video_detector.m

body_detector.o: body_detector.m body_detector.h face_detector_internal.h
	$(CC) $(OBJCFLAGS) -c body_detector.m

clean:
	rm -f $(OBJECTS) $(TARGET)
