#pragma once

#include <SFML/Graphics.hpp>

class Drawable {
    virtual void draw(sf::RenderWindow &) {}
};