#include <iostream>

#include <SFML/Graphics.hpp>

#include "Food.hpp"

Food::Food() {
    shape = sf::CircleShape(SIZE);
    shape.setFillColor(COLOR);
    shape.setOrigin(SIZE, SIZE);
}

void Food::draw(sf::RenderWindow &window) {
    window.draw(shape);
}

void Food::move(sf::RenderWindow &window) {
    // top collision
    if (shape.getPosition().y - SIZE <= 0)
        mvmntVec = *(new sf::Vector2f(mvmntVec.x, -mvmntVec.y));
    // bottom collision
    if (shape.getPosition().y + SIZE >= window.getSize().y)
        mvmntVec = *(new sf::Vector2f(mvmntVec.x, -mvmntVec.y));
    // left collision
    if (shape.getPosition().x - SIZE <= 0)
        mvmntVec = *(new sf::Vector2f(-mvmntVec.x, mvmntVec.y));
    // right collision
    if (shape.getPosition().x + SIZE >= window.getSize().x)
        mvmntVec = *(new sf::Vector2f(-mvmntVec.x, mvmntVec.y));

    shape.move(mvmntVec);
}

void Food::reset(sf::Vector2f &pos, sf::Vector2f &mvmntVec_) {
    shape.setPosition(pos);
    mvmntVec = mvmntVec_; 
}

sf::Vector2f Food::getPos() {
    return shape.getPosition();
}