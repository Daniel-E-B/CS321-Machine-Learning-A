#include <SFML/Graphics.hpp>

#include "Food.hpp"

Food::Food(sf::Vector2f &pos, sf::Vector2f &mvmntVec_) {
    shape = sf::CircleShape(SIZE);
    shape.setFillColor(COLOR);
    shape.setOrigin(SIZE, SIZE);
    shape.setPosition(pos);
    mvmntVec = mvmntVec_;
}

void Food::draw(sf::RenderWindow &window) {
    window.draw(shape);
}

void Food::move() {
    shape.move(mvmntVec);
}