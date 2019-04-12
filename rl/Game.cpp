#include <cmath>
#include <ctime>
#include <random>
#include <thread>

#include <SFML/Graphics.hpp>
#include <nlohmann/json.hpp>

#include "Basket.hpp"
#include "Food.hpp"
#include "Game.hpp"

void Game::reset(sf::RenderWindow &window) {
    std::mt19937_64 re(std::time(0));
    std::uniform_real_distribution<double> velo(-MAX_SPEED_COMPONENT, MAX_SPEED_COMPONENT);
    std::uniform_real_distribution<double> x(0, window.getSize().x);
    std::uniform_real_distribution<double> y(0, window.getSize().y);

    basket->setPos(*(new sf::Vector2f(x(re), y(re))));
    food->reset(*(new sf::Vector2f(x(re), y(re))), *(new sf::Vector2f(velo(re), velo(re))));
}

Game::Game(sf::RenderWindow &window) {
    fitnesses.resize(CREATURES);
    brains.resize(CREATURES);
    food = new Food();
    basket = new Basket();
    for (int i = 0; i < CREATURES; ++i) {
        brains[i] = basket->getBrain();
    }
    mutate();
    reset(window);
}

void Game::draw(sf::RenderWindow &window) {
    food->draw(window);
    basket->draw(window);
}

void Game::tick(sf::RenderWindow &window) {
    food->move(window);
    basket->move(*(new sf::Vector2f(food->getPos().x / window.getSize().x, food->getPos().y / window.getSize().y)));
}

double Game::fitness() {
    return (sqrt(pow(food->getPos().x - basket->getPos().x, 2) + pow(food->getPos().y - basket->getPos().y, 2)));
}

void Game::mutate() {
    std::mt19937_64 re(std::time(0));
    std::uniform_real_distribution<double> dist(-1, 1);
    bool flag = true;
    while (flag) {
        flag = false;
        for (int i = 1; i < fitnesses.size(); ++i) {
            if (fitnesses[i] < fitnesses[i - 1]) {
                flag = true;
                double tmpF = fitnesses[i];
                fitnesses[i] = fitnesses[i - 1];
                fitnesses[i - 1] = tmpF;
                std::string tmpB = brains[i];
                brains[i] = brains[i - 1];
                brains[i - 1] = tmpB;
            }
        }
    }
}

void Game::generation(sf::RenderWindow &window, unsigned long long int tickFreq, bool &stop) {
    for (int i = 0; i < CREATURES; ++i) {
        reset(window);
        basket->setBrain(brains[i]);
        fitnesses[i] = 0;
        for (int j = 0; j < TICKS_PER_GENERATION; ++j) {
            fitnesses[i] += fitness();
            tick(window);
            if (stop) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(tickFreq));
        }
    }
    mutate();
}