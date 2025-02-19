# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import importlib.util
from typing import Any, Callable, Optional, Union

import numpy as np

if importlib.util.find_spec("pygame") is None:
    import pip

    pip.main(["install", "pygame==2.1.2"])
import pygame


class PyGameRenderer:
    """This is a generic pygame runner that allows user to
    provide custom rendering function and keybinding functions."""

    def __init__(
        self,
        teleop: bool = False,
        height: int = 400,
        width: int = 600,
        fps: Union[float, int] = 60,
        render_func: Optional[Callable[[], np.ndarray]] = None,
        step_func: Optional[Callable[[], None]] = None,
        key_func: Optional[Callable[[int], Any]] = None,
    ):
        self.img_width = width
        self.img_height = height

        self.key_func = key_func
        self.render_func = render_func
        self.step_func = step_func

        self.teleop = teleop
        self.running = False
        self.fps = fps

    def on_init(self) -> None:
        pygame.init()
        # init main screen and background
        self._display_surf = pygame.display.set_mode((self.img_width, self.img_height), pygame.HWSURFACE)
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()

        # Font
        self._myfont = pygame.font.SysFont("Comic Sans MS", 30)
        self._running = True

    def on_cleanup(self) -> None:
        pygame.display.quit()
        pygame.quit()

    def on_execute(self) -> None:
        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

    def on_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            if not self.teleop:
                return
            if self.key_func is not None:
                self.key_func(event.key)

    def on_loop(self) -> None:
        if self.step_func is not None:
            self.step_func()
        self._clock.tick(self.fps)

    def render(self) -> np.ndarray:
        """Returns an image that visualizes the state"""
        if self.render_func is not None:
            img = self.render_func()
        else:
            img = np.zeros((self.img_width, self.img_height))
        return img

    def on_render(self) -> None:
        # Note: pygame 2.5.2 (newest) fails displaying the image somehow; had to downgrade pygame (old version 2.1.0).
        img = self.render()
        img = np.transpose(img, (1, 0, 2))
        fps_text = "FPS: {0:.2f}".format(self._clock.get_fps())
        pygame.surfarray.blit_array(self._display_surf, img)
        pygame.display.set_caption(f"{fps_text}")
        pygame.display.flip()
