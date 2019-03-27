#pragma once

#include <SFML/Graphics.hpp>
#include "GameObject.hpp"

class Player : public GameObject {
   private:
    sf::CircleShape shape;
    const sf::Color COLOR = sf::Color(0, 127, 127);
    const double SIZE = 50;

   public:
    Player(sf::Vector2f &);
    void draw(sf::RenderWindow &);
    void move(sf::Vector2f &);
};