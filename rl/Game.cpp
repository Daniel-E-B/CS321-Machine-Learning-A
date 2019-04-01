#include <random>

#include <SFML/Graphics.hpp>

#include "Basket.hpp"
#include "Food.hpp"
#include "Game.hpp"

Game::Game(sf::RenderWindow &window) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-MAX_SPEED_COMPONENT, MAX_SPEED_COMPONENT);

    food = new Food(*(new sf::Vector2f(window.getSize().x / 2, window.getSize().y / 2)), *(new sf::Vector2f(dist(mt), (dist(mt)))));
    basket = new Basket(*(new sf::Vector2f(window.getSize().x / 2, window.getSize().y / 2)));
}

void Game::draw(sf::RenderWindow &window) {
    food->draw(window);
    basket->draw(window);
}

void Game::setFood(sf::Vector2f &pos, sf::Vector2f &mvmntVec) {
}

bool Game::tick(sf::RenderWindow &window) {
    food->move(window);
    basket->move(*(new sf::Vector2f(food->getPos().x * MAX_SPEED_COMPONENT, food->getPos().y * MAX_SPEED_COMPONENT)));
}