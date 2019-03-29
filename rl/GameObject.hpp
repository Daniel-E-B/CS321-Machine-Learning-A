#pragma once

#include <SFML/Graphics.hpp>

#include "Drawable.hpp"

class GameObject : public Drawable{
   public:
    virtual void move(sf::Vector2f &) {}
    // TODO: implement rotate (maybe just for the basket? (cuz circles dont need to rotate))
};