#pragma once

#include <SFML/Graphics.hpp>

class GameObject {
   public:
    virtual void draw(sf::RenderWindow &) {}
    virtual void move(sf::Vector2f &) {}
};